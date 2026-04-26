from huggingface_hub import HfApi

# ── CHANGE THIS ──────────────────────────
HF_USERNAME = "anikeshroyy"   # ← your HF username
SPACE_NAME  = "resume-parser"
# ─────────────────────────────────────────

REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"
api     = HfApi()

print("Uploading app.py...")
api.upload_file(
    path_or_fileobj = "app.py",
    path_in_repo    = "app.py",
    repo_id         = REPO_ID,
    repo_type       = "space"
)

print("Uploading Dockerfile...")
api.upload_file(
    path_or_fileobj = "Dockerfile",
    path_in_repo    = "Dockerfile",
    repo_id         = REPO_ID,
    repo_type       = "space"
)

print("Uploading requirements.txt...")
api.upload_file(
    path_or_fileobj = "requirements.txt",
    path_in_repo    = "requirements.txt",
    repo_id         = REPO_ID,
    repo_type       = "space"
)

print("Uploading README.md...")
api.upload_file(
    path_or_fileobj = "README.md",
    path_in_repo    = "README.md",
    repo_id         = REPO_ID,
    repo_type       = "space"
)

print("Uploading model folder (takes 2-3 mins)...")
api.upload_folder(
    folder_path  = "model/resume_ner",
    path_in_repo = "model/resume_ner",
    repo_id      = REPO_ID,
    repo_type    = "space"
)

print("\nAll uploaded!")
print(f"Space URL: https://huggingface.co/spaces/{REPO_ID}")