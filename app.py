import streamlit as st
import pdfplumber
import io
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- HELPER: EXTRACT TEXT FROM PDF -----------------
def extract_text_from_pdf(uploaded_file):
    if uploaded_file is None:
        return ""
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ----------------- HELPER: CLEAN TEXT -----------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------- SKILLS LIST (Expanded) -----------------
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "html", "css", "javascript", "typescript",
    "machine learning", "deep learning", "nlp", "data analysis", "ai", "analysis",
    "excel", "power bi", "tableau", "data science", "statistics",
    "django", "flask", "react", "angular", "node",
    "rest api", "git", "github", "docker", "aws", "cloud",
    "communication", "teamwork", "leadership", "problem solving",
    "time management", "project management", "critical thinking"
]

def extract_skills(text: str, skill_keywords):
    text_low = text.lower()
    present = []
    for skill in skill_keywords:
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_low):
            present.append(skill)
    return sorted(list(set(present)))

# ----------------- CORE LOGIC: ANALYSIS -----------------
def analyze_resume(resume_text: str, job_description: str):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([jd_clean, resume_clean])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_score = int(round(similarity * 100))

    jd_skills = extract_skills(job_description, SKILL_KEYWORDS)
    resume_skills = extract_skills(resume_text, SKILL_KEYWORDS)

    strengths = [s for s in jd_skills if s in resume_skills]
    gaps = [s for s in jd_skills if s not in resume_skills]

    if match_score >= 75:
        decision = "Shortlist"
    elif match_score >= 50:
        decision = "Maybe"
    else:
        decision = "Reject"

    summary = (
        f"The resume shows ability in {', '.join(strengths) or 'relevant areas'} "
        f"but lacks clear proof of {', '.join(gaps) or 'a few important skills'}. "
        f"Overall Fit: {decision} ({match_score}/100)"
    )

    return match_score, decision, strengths, gaps, summary, jd_skills, resume_skills


# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="AI Resume Screening Agent", page_icon="📄", layout="wide")

st.markdown("<h1 style='text-align: center;'>📄 AI Resume Screening Agent</h1>", unsafe_allow_html=True)
st.write("This tool analyzes your resume against a Job Description and provides insights like match score, skill strengths and missing skills.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📌 Upload Resume (PDF Only)", type=["pdf"])
with col2:
    job_description = st.text_area("📝 Paste Job Description Here", height=200)

st.markdown("---")

if st.button("🚀 Start Analysis", use_container_width=True):
    if uploaded_file and job_description.strip():
        with st.spinner("Analyzing Resume... Please wait! ⏳"):
            resume_text = extract_text_from_pdf(uploaded_file)
            match_score, decision, strengths, gaps, summary, jd_skills, resume_skills = analyze_resume(resume_text, job_description)

        st.success("Analysis Completed Successfully 🎯")

        st.metric("Match Score", f"{match_score} / 100")
        st.write(f"### Final Decision: **{decision}**")

        st.write("### Summary")
        st.info(summary)

        st.write("### Strengths (Skills Present)")
        if strengths:
            st.success(", ".join(strengths))
        else:
            st.write("No major strengths detected.")

        st.write("### Skill Gaps (Missing from Resume)")
        if gaps:
            st.error(", ".join(gaps))
        else:
            st.write("No major gaps found.")

        with st.expander("Detailed Skill Match Breakdown"):
            st.write("**Skills in Job Description:**")
            st.write(", ".join(jd_skills) or "None")
            st.write("**Skills found in Resume:**")
            st.write(", ".join(resume_skills) or "None")

else:
    st.warning("Please upload a resume and paste a Job Description to begin analysis.")

