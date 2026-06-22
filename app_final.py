import gradio as gr
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    "python", "java", "javascript", "typescript", "sql", "go", "r",
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "pandas", "numpy", "matplotlib", "scikit-learn",
    "machine learning", "deep learning", "nlp", "computer vision", "data science",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "mongodb", "postgresql", "mysql", "redis",
    "agile", "scrum", "leadership", "communication", "teamwork",
    "html", "css", "rest api", "microservices", "tableau", "power bi",
]

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_skills(text):
    text_lower = text.lower()
    return {s for s in SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text_lower)}

def analyze(resume_pdf, jd_text):
    if resume_pdf is None:
        return "Please upload a resume PDF.", "", "", ""
    if not jd_text or not jd_text.strip():
        return "Please paste a job description.", "", "", ""

    resume_text = extract_text(resume_pdf)
    if not resume_text.strip():
        return "Could not extract text from PDF.", "", "", ""

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    score = round(float(cosine_similarity(tfidf[0], tfidf[1])[0][0]) * 100, 1)

    if score >= 75:
        label = "Strong Match ✅"
    elif score >= 50:
        label = "Moderate Match ⚠️"
    else:
        label = "Weak Match ❌"

    r_skills = extract_skills(resume_text)
    j_skills = extract_skills(jd_text)
    matched = ", ".join(sorted(r_skills & j_skills)) or "None detected"
    missing = ", ".join(sorted(j_skills - r_skills)) or "None — great fit!"

    tips = []
    if j_skills - r_skills:
        tips.append(f"• Add these keywords: {', '.join(list(j_skills - r_skills)[:5])}")
    if score < 50:
        tips.append("• Tailor your resume specifically to this job description.")
    if score >= 75:
        tips.append("• Strong match! Use similar language to the JD.")
    tips.append("• Quantify achievements (e.g. 'Improved accuracy by 15%')")

    return f"{score}% — {label}", matched, missing, "\n".join(tips)

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="Upload Resume PDF", file_types=[".pdf"]),
        gr.Textbox(label="Paste Job Description", lines=8, placeholder="Paste job description here...")
    ],
    outputs=[
        gr.Textbox(label="Match Score"),
        gr.Textbox(label="Matched Skills"),
        gr.Textbox(label="Missing Skills"),
        gr.Textbox(label="Tips to Improve"),
    ],
    title="🎯 Resume Job Matcher",
    description="Upload your resume and paste a job description to see how well you match."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
