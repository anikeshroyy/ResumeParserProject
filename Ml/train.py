import json
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────
DATASET_PATH = "data/Entity Recognition in Resumes.json"
MODEL_PATH   = "model/resume_ner"
BASE_MODEL   = "en_core_web_sm"
EPOCHS       = 50
RANDOM_SEED  = 42

# ── LOAD DATA ────────────────────────────────────────────────────────
def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} resumes")
    return data

# ── CONVERT TO SPACY FORMAT ──────────────────────────────────────────
def convert_to_spacy(data):
    training_data = []
    skipped = 0
    for entry in data:
        text        = entry.get("content", "").strip()
        annotations = entry.get("annotation") or []
        if not text or not annotations:
            skipped += 1
            continue
        entities = []
        for ann in annotations:
            label_list = ann.get("label", [])
            if not label_list:
                continue
            label = label_list[0].upper()
            for point in ann.get("points", []):
                start = point.get("start")
                end   = point.get("end")
                if start is None or end is None:
                    continue
                end = end + 1
                if start < 0 or end > len(text) or start >= end:
                    continue
                entities.append((start, end, label))
        if entities:
            training_data.append((text, {"entities": entities}))
        else:
            skipped += 1
    print(f"Valid: {len(training_data)} | Skipped: {skipped}")
    return training_data

# ── CLEAN ENTITIES ───────────────────────────────────────────────────
def clean_entities(training_data, nlp):
    clean_data    = []
    total_removed = 0
    for text, annotations in training_data:
        valid_entities = []
        for start, end, label in annotations["entities"]:
            if start < 0 or end > len(text) or start >= end:
                total_removed += 1
                continue
            span_text = text[start:end]
            stripped  = span_text.strip()
            if not stripped:
                total_removed += 1
                continue
            leading   = len(span_text) - len(span_text.lstrip())
            trailing  = len(span_text) - len(span_text.rstrip())
            new_start = start + leading
            new_end   = end   - trailing
            if new_start >= new_end:
                total_removed += 1
                continue
            doc  = nlp.make_doc(text)
            span = doc.char_span(new_start, new_end, label=label,
                                  alignment_mode="expand")
            if span is None:
                total_removed += 1
                continue
            valid_entities.append((span.start_char, span.end_char, label))

        valid_entities  = sorted(valid_entities, key=lambda x: (x[0], -(x[1]-x[0])))
        non_overlapping = []
        last_end        = -1
        for start, end, label in valid_entities:
            if start >= last_end:
                non_overlapping.append((start, end, label))
                last_end = end
        if non_overlapping:
            clean_data.append((text, {"entities": non_overlapping}))

    print(f"Clean: {len(clean_data)} | Removed: {total_removed}")
    return clean_data

# ── AUGMENT DATA ─────────────────────────────────────────────────────
def augment_data(training_data, multiply=3):
    augmented = []
    for text, annotations in training_data:
        augmented.append((text, annotations))
        for _ in range(multiply - 1):
            new_text = text
            lines    = new_text.split("\n")
            if len(lines) > 3:
                insert_pos = random.randint(1, len(lines) - 1)
                lines.insert(insert_pos, "")
                new_text = "\n".join(lines)
            new_entities = []
            for start, end, label in annotations["entities"]:
                span      = text[start:end]
                new_start = new_text.find(span)
                if new_start != -1:
                    new_end = new_start + len(span)
                    if new_text[new_start:new_end] == span:
                        new_entities.append((new_start, new_end, label))
            if new_entities:
                augmented.append((new_text, {"entities": new_entities}))
    print(f"Augmented: {len(augmented)} samples")
    return augmented

# ── TRAIN ────────────────────────────────────────────────────────────
def train(train_data, test_data):
    nlp = spacy.load(BASE_MODEL)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, ann in train_data:
        for (_, _, label) in ann["entities"]:
            ner.add_label(label)

    other_pipes    = [p for p in nlp.pipe_names if p != "ner"]
    losses_history = []
    best_loss      = float("inf")
    patience       = 10
    no_improve     = 0

    Path("model").mkdir(exist_ok=True)

    print(f"\nTraining on {len(train_data)} samples for {EPOCHS} epochs...\n")

    with nlp.disable_pipes(*other_pipes):
        optimizer            = nlp.resume_training()
        optimizer.learn_rate = 0.001

        for epoch in range(EPOCHS):
            random.shuffle(train_data)
            losses  = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                examples = []
                for text, annotations in batch:
                    try:
                        doc     = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    except Exception:
                        continue
                if not examples:
                    continue
                try:
                    dropout = max(0.1, 0.5 - (epoch * 0.004))
                    nlp.update(examples, drop=dropout, losses=losses)
                except Exception:
                    continue

            loss = round(losses.get("ner", 0), 3)
            losses_history.append(loss)

            if loss < best_loss:
                best_loss  = loss
                no_improve = 0
                nlp.to_disk(MODEL_PATH)
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:03d}/{EPOCHS} — Loss: {loss:.3f} — Best: {best_loss:.3f}")

            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete! Best loss: {best_loss:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    return losses_history

# ── EVALUATE ─────────────────────────────────────────────────────────
def evaluate(test_data):
    nlp           = spacy.load(MODEL_PATH)
    test_examples = []
    for text, annotations in test_data:
        doc     = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        test_examples.append(example)

    scores = nlp.evaluate(test_examples)
    print("\n" + "=" * 45)
    print("       EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Precision : {scores['ents_p']:.4f}")
    print(f"  Recall    : {scores['ents_r']:.4f}")
    print(f"  F1 Score  : {scores['ents_f']:.4f}")
    print("=" * 45)
    print(f"\n  {'Label':<30} {'P':>6} {'R':>6} {'F1':>6}")
    print("  " + "-" * 46)
    for label, m in sorted(scores.get("ents_per_type", {}).items()):
        print(f"  {label:<30} {m['p']:>6.3f} {m['r']:>6.3f} {m['f']:>6.3f}")

# ── PLOT LOSS ────────────────────────────────────────────────────────
def plot_loss(losses_history):
    plt.figure(figsize=(9, 4))
    plt.plot(losses_history, marker="o", markersize=3,
             linewidth=1.8, color="steelblue")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("NER Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model/training_loss.png", dpi=150)
    plt.show()
    print("Loss curve saved to model/training_loss.png")

# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    # 1. Load
    raw_data   = load_data(DATASET_PATH)

    # 2. Prepare
    base_nlp   = spacy.load(BASE_MODEL)
    train_data = convert_to_spacy(raw_data)
    train_data = clean_entities(train_data, base_nlp)

    # 3. Split
    random.shuffle(train_data)
    split      = int(0.8 * len(train_data))
    test_data  = train_data[split:]
    train_data = train_data[:split]

    # 4. Augment
    train_data = augment_data(train_data, multiply=3)

    # 5. Train
    losses = train(train_data, test_data)

    # 6. Evaluate
    evaluate(test_data)

    # 7. Plot
    plot_loss(losses)