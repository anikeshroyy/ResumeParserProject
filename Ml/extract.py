import spacy
import pdfplumber
import re
import json
from pathlib import Path

MODEL_PATH = "model/resume_ner"

# Load model
nlp = spacy.load(MODEL_PATH)

KNOWN_SKILLS = [
    ("python","Python"), ("java","Java"),
    ("javascript","JavaScript"), ("typescript","TypeScript"),
    ("c++","C++"), ("c#","C#"), ("golang","Go"),
    ("html5","HTML5"), ("html","HTML"), ("css3","CSS3"), ("css","CSS"),
    ("react.js","React.js"), ("react","React"),
    ("next.js","Next.js"), ("vue","Vue"), ("angular","Angular"),
    ("tailwind css","Tailwind CSS"), ("tailwind","Tailwind CSS"),
    ("bootstrap","Bootstrap"), ("node.js","Node.js"),
    ("express.js","Express.js"), ("django","Django"),
    ("flask","Flask"), ("fastapi","FastAPI"),
    ("mysql","MySQL"), ("postgresql","PostgreSQL"),
    ("mongodb","MongoDB"), ("redis","Redis"),
    ("firebase","Firebase"), ("aws","AWS"), ("azure","Azure"),
    ("docker","Docker"), ("kubernetes","Kubernetes"),
    ("git","Git"), ("github","GitHub"), ("bitbucket","Bitbucket"),
    ("linux","Linux"), ("postman","Postman"),
    ("vs code","VS Code"), ("anaconda","Anaconda"),
    ("tensorflow","TensorFlow"), ("pytorch","PyTorch"),
    ("opencv","OpenCV"), ("machine learning","Machine Learning"),
    ("deep learning","Deep Learning"), ("nlp","NLP"),
    ("rest api","REST API"), ("graphql","GraphQL"),
    ("dsa","DSA"), ("oops","OOP"), ("dbms","DBMS"),
    ("operating system","Operating System"),
    ("computer networks","Computer Networks"),
    ("c programming","C Programming"),
    ("embedded c","Embedded C"), ("iot","IoT"),
    ("esp32","ESP32"), ("ci/cd","CI/CD"), ("agile","Agile"),
    ("figma","Figma"), ("canva","Canva"),
    ("flutter","Flutter"), ("dart","Dart"),
    ("kotlin","Kotlin"), ("swift","Swift"),
    ("sql","SQL"), ("sqlite","SQLite"),
    ("mapbox","Mapbox"), ("smtp","SMTP"),
    ("can protocol","CAN Protocol"), ("embedded c","Embedded C"),
    ("tcp/ip","TCP/IP"), ("bitbucket","Bitbucket"),
]

COMPANY_KEYWORDS = [
    "motor","company","pvt","ltd","inc","corp","technologies",
    "solutions","services","software","systems","it ",
    "marketplace","ventures","enterprises","consulting","labs"
]

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(
            page.extract_text() for page in pdf.pages
            if page.extract_text()
        )

def extract_email(text):
    return list(set(re.findall(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text
    )))

def extract_phone(text):
    m = re.findall(r"(\+91[\-\s]?\d{10}|\b[6-9]\d{9}\b)", text)
    return list(set(x.strip() for x in m if len(re.sub(r"\D", "", x)) >= 10))

def extract_cgpa(text):
    m = re.search(r"cgpa[\s:\-]*([0-9]\.[0-9]+)", text, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r"([0-9]\.[0-9]+)\s*/\s*10", text)
    if m: return m.group(1)
    return None

def extract_graduation_year(text):
    m = re.search(r"20(2[0-9])\s*[-–]\s*20(2[0-9]|3[0-5])", text)
    return m.group(0) if m else None

def extract_experience(text):
    # Explicit years
    m = re.search(
        r"(\d+\.?\d*)\+?\s*years?\s+(?:of\s+)?experience",
        text, re.IGNORECASE
    )
    if m: return m.group(0)

    # Only check experience/internship section for "Present"
    exp_section = re.search(
        r"(experience|internship|employment)"
        r"(.*?)"
        r"(?=\n(?:education|skills|projects|certifications|extracurricular|coding)|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if exp_section:
        s = exp_section.group(2)
        m = re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
            r"\s+\d{4}\s*[-–]\s*(present|current)",
            s, re.IGNORECASE
        )
        if m: return f"Experienced (since {m.group(0)})"
        if re.search(r"intern", s, re.IGNORECASE):
            return "Fresher / Intern"

    if re.search(r"(fresher|pursuing)", text, re.IGNORECASE):
        return "Fresher / Entry Level"
    if re.search(r"b\.?tech", text, re.IGNORECASE):
        return "Fresher / Entry Level"
    return None

def extract_skills(text):
    found      = {}
    text_lower = text.lower()
    for keyword, display in KNOWN_SKILLS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
            found[keyword.lower()] = display
    # Remove generic when specific exists
    for generic, specific in [
        ("react",   "react.js"),
        ("tailwind","tailwind css"),
        ("html",    "html5"),
        ("css",     "css3"),
    ]:
        if generic in found and specific in found:
            del found[generic]
    return sorted(found.values())

def extract_projects(text):
    projects = []
    seen     = set()
    lines    = text.split("\n")

    # Section headers to never treat as project names
    SECTION_HEADERS = {
        "experience", "education", "skills", "projects", "certifications",
        "achievements", "extracurricular", "summary", "objective", "profile",
        "profile summary", "technical skills", "coursework", "internship",
        "coding platforms", "languages", "tools", "frameworks", "contact"
    }

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip section headers
        if line.lower().strip() in SECTION_HEADERS:
            continue

        # Pattern 1: "ProjectName | Live | Github"
        m = re.match(r"^([A-Z][A-Za-z0-9\s\-]{2,35})\s*\|\s*Live\s*\|", line)
        if m:
            name = m.group(1).strip()
            if name.lower() in SECTION_HEADERS:
                continue
            tech = ""
            for j in range(i+1, min(i+6, len(lines))):
                tech_match = re.search(
                    r"(?:Developed using|Tech(?:nical)?\s*Stack[:\-\s]+|Tech\-Stacks?[:\-\s]+)(.+)",
                    lines[j], re.IGNORECASE
                )
                if tech_match:
                    tech = tech_match.group(1).strip()
                    break
            if name not in seen:
                projects.append({"name": name, "tech_stack": tech or "See resume"})
                seen.add(name)
            continue

        # Pattern 2: "ProjectName W | Next.js, React, Tailwind CSS"
        m2 = re.match(
            r"^([A-Z][A-Za-z0-9\s\-]{2,35})\s*(?:W\s*)?\|\s*"
            r"((?:(?:React|Node|HTML|CSS|JavaScript|Python|MongoDB|Express"
            r"|Tailwind|Bootstrap|Django|Flask|AWS|Firebase|Next\.js|Vue"
            r"|Angular|Mapbox|MySQL|PostgreSQL|Redis|Docker|Git|Flutter"
            r"|Dart|TypeScript|Kotlin|Swift|Spring|FastAPI|IDE|VS\s*Code"
            r")[,\.\s\-]*)+)",
            line
        )
        if m2:
            name  = m2.group(1).strip()
            stack = m2.group(2).strip(" ,")
            if name.lower() not in SECTION_HEADERS and name not in seen and len(name) > 3:
                projects.append({"name": name, "tech_stack": stack})
                seen.add(name)
            continue

        # Pattern 3: Standalone capitalized project name
        # Must NOT be a section header and next line must be a bullet
        m3 = re.match(r"^([A-Z][A-Za-z0-9\s\-]{2,40})$", line)
        if m3:
            name = m3.group(1).strip()
            if name.lower() in SECTION_HEADERS:
                continue
            if i+1 < len(lines) and lines[i+1].strip().startswith("•"):
                tech = ""
                for j in range(i+1, min(i+6, len(lines))):
                    tech_match = re.search(
                        r"(?:using|built with|stack|technologies)[:\s]+(.+)",
                        lines[j], re.IGNORECASE
                    )
                    if tech_match:
                        tech = tech_match.group(1).strip()
                        break
                if name not in seen:
                    projects.append({"name": name, "tech_stack": tech or "See resume"})
                    seen.add(name)

    return projects
def extract_certifications(text):
    certs = []
    section = re.search(
        r"certifications?\s*\n(.*?)(?=\n[A-Z]{3,}|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if section:
        for line in section.group(1).strip().split("\n"):
            line = line.strip(" •-–*·")
            if len(line) > 10:
                certs.append(line)
    return certs

def detect_profile(text):
    t = text.lower()

    scores = {
        "ML / Data Science"      : 0,
        "Embedded / IoT Engineer": 0,
        "Mobile Developer"       : 0,
        "Full Stack Developer"   : 0,
        "Backend Developer"      : 0,
        "Frontend Developer"     : 0,
    }

    # ML / Data Science
    for kw in ["machine learning","deep learning","tensorflow","pytorch",
               "nlp","data science","scikit","computer vision","keras"]:
        if kw in t: scores["ML / Data Science"] += 3

    # Embedded — strong signals only
    for kw in ["embedded c","microcontroller","firmware","rtos",
               "can protocol","ota","telematics","misra","ecu"]:
        if kw in t: scores["Embedded / IoT Engineer"] += 3
    # Weak IoT signals
    for kw in ["iot","esp32","arduino","raspberry pi","sensor"]:
        if kw in t: scores["Embedded / IoT Engineer"] += 1

    # Mobile
    for kw in ["flutter","dart","android","ios","react native",
               "kotlin","swift","mobile app","play store"]:
        if kw in t: scores["Mobile Developer"] += 3

    # Full Stack
    for kw in ["full stack","fullstack","mern","mean"]:
        if kw in t: scores["Full Stack Developer"] += 4

    # Backend
    for kw in ["node.js","express.js","django","flask","fastapi",
               "microservices","rest api","backend","api development"]:
        if kw in t: scores["Backend Developer"] += 2

    # Frontend
    for kw in ["react","next.js","vue","angular","html5","css3",
               "tailwind","bootstrap","frontend","web development"]:
        if kw in t: scores["Frontend Developer"] += 2

    # Boost from experience/internship section
    exp_section = re.search(
        r"(experience|internship)(.*?)(?=\n(?:education|skills|projects)|$)",
        t, re.DOTALL
    )
    if exp_section:
        exp_text = exp_section.group(2)
        if any(x in exp_text for x in ["react","next.js","frontend","html","tailwind"]):
            scores["Frontend Developer"] += 3
        if any(x in exp_text for x in ["node","express","backend","api","django"]):
            scores["Backend Developer"] += 3
        if any(x in exp_text for x in ["flutter","android","mobile","dart"]):
            scores["Mobile Developer"] += 3
        if any(x in exp_text for x in ["embedded c","firmware","can protocol","telematics"]):
            scores["Embedded / IoT Engineer"] += 3
        if any(x in exp_text for x in ["machine learning","tensorflow","data science"]):
            scores["ML / Data Science"] += 3

    # If strong both frontend and backend → Full Stack
    if scores["Frontend Developer"] >= 4 and scores["Backend Developer"] >= 4:
        scores["Full Stack Developer"] += (
            scores["Frontend Developer"] + scores["Backend Developer"]
        )

    scores = {k: v for k, v in scores.items() if v > 0}
    if not scores:
        return "Software Developer"

    return max(scores, key=scores.get)

def is_company(name):
    return any(kw in name.lower() for kw in COMPANY_KEYWORDS)

def parse_resume(pdf_path):
    print(f"\nParsing: {pdf_path}")

    raw_text   = extract_text_from_pdf(pdf_path)
    doc        = nlp(raw_text)
    ner_result = {}

    for ent in doc.ents:
        clean = re.sub(r"\s{2,}", " ", ent.text.strip().replace("\n", " "))
        if clean:
            ner_result.setdefault(ent.label_, []).append(clean)

    # Deduplicate NER
    for label in ner_result:
        seen = []
        for v in ner_result[label]:
            if v not in seen:
                seen.append(v)
        ner_result[label] = seen

    result = {
        "name"          : ner_result.get("NAME", []),
        "email"         : extract_email(raw_text),
        "phone"         : extract_phone(raw_text),
        "linkedin"      : list(set(re.findall(
                            r"linkedin\.com/in/[\w\-]+", raw_text, re.IGNORECASE))),
        "github"        : list(set(re.findall(
                            r"github\.com/[\w\-]+", raw_text, re.IGNORECASE))),
        "college"       : [v for v in ner_result.get("COLLEGE NAME", [])
                           if not is_company(v)],
        "degree"        : ner_result.get("DEGREE", []),
        "cgpa"          : extract_cgpa(raw_text),
        "grad_year"     : extract_graduation_year(raw_text),
        "experience"    : extract_experience(raw_text),
        "profile_type"  : detect_profile(raw_text),
        "companies"     : [v for v in ner_result.get("COMPANIES WORKED AT", [])
                           if is_company(v)],
        "designation"   : ner_result.get("DESIGNATION", []),
        "skills"        : extract_skills(raw_text),
        "projects"      : extract_projects(raw_text),
        "certifications": extract_certifications(raw_text),
    }

    result = {k: v for k, v in result.items() if v}
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "resumes/test.pdf"

    result = parse_resume(pdf_path)

    print("\n" + "=" * 50)
    print("    EXTRACTED RESUME INFORMATION")
    print("=" * 50)
    for key, value in result.items():
        print(f"\n{key.upper()}:")
        if key == "projects":
            for p in value:
                print(f"  • {p['name']}  |  {p['tech_stack']}")
        elif isinstance(value, list):
            for v in value:
                print(f"  • {v}")
        else:
            print(f"  • {value}")

    output_file = "extracted_resume.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_file}")