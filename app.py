import streamlit as st
import pandas as pd
import numpy as np
import re
import hashlib
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🤖 AI-Powered Resume Screening System")

# -------------------------------
# MONGODB CONNECTION
# -------------------------------
MONGO_URI = st.secrets["MONGO_URI"]

client = MongoClient(MONGO_URI)
db = client["resume_analyzer"]
users_collection = db["users"]
results_collection = db["results"]

# -------------------------------
# SESSION STATE
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------
# PASSWORD HASH
# -------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------------------
# SKILL DICTIONARY
# -------------------------------
SKILLS = [
    "python", "java", "sql", "machine learning",
    "deep learning", "nlp", "html", "css",
    "javascript", "mongodb", "react", "flask",
    "django", "aws", "excel"
]

# -------------------------------
# EXPERIENCE EXTRACTION
# -------------------------------
def extract_experience(text):
    pattern = r"(\d+)\+?\s*(years|year)"
    matches = re.findall(pattern, text.lower())
    years = [int(m[0]) for m in matches]
    return max(years) if years else 0

# -------------------------------
# SKILL EXTRACTION
# -------------------------------
def extract_skills(text):
    text = text.lower()
    found = [skill for skill in SKILLS if skill in text]
    return found

# -------------------------------
# LOGIN / REGISTER
# -------------------------------
if not st.session_state.logged_in:

    st.subheader("🔐 Login / Register")
    choice = st.radio("Choose Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Register":
        if st.button("Register"):
            if users_collection.find_one({"username": username}):
                st.error("User already exists")
            else:
                users_collection.insert_one({
                    "username": username,
                    "password": hash_password(password)
                })
                st.success("Registration Successful! Please Login.")

    else:
        if st.button("Login"):
            user = users_collection.find_one({
                "username": username,
                "password": hash_password(password)
            })
            if user:
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

# -------------------------------
# MAIN APP
# -------------------------------
else:

    st.sidebar.success("✅ Logged In")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.sidebar.markdown("## 📊 Admin Panel")

    if st.sidebar.button("View All Results"):
        data = list(results_collection.find({}, {"_id": 0}))
        if data:
            df_admin = pd.DataFrame(data)
            st.subheader("📁 All Stored Results")
            st.dataframe(df_admin)
        else:
            st.info("No results in database.")

    st.markdown("---")

    st.subheader("📄 Enter Job Description")
    jd_text = st.text_area("Paste Job Description Here")

    uploaded_files = st.file_uploader(
        "Upload Resume Files (.txt only)",
        accept_multiple_files=True
    )

    if st.button("Analyze Resumes"):

        if not jd_text or not uploaded_files:
            st.warning("Please upload resumes and enter job description.")
        else:

            with st.spinner("Analyzing resumes with AI..."):

                vectorizer = TfidfVectorizer()
                documents = [jd_text]

                resume_texts = []

                for file in uploaded_files:
                    text = file.read().decode("utf-8")
                    resume_texts.append((file.name, text))
                    documents.append(text)

                tfidf_matrix = vectorizer.fit_transform(documents)
                jd_vector = tfidf_matrix[0]

                results = []

                for i, (filename, text) in enumerate(resume_texts):

                    resume_vector = tfidf_matrix[i + 1]
                    similarity_score = cosine_similarity(
                        jd_vector, resume_vector
                    )[0][0]

                    skills_found = extract_skills(text)
                    skill_score = len(skills_found) / len(SKILLS)

                    experience_years = extract_experience(text)

                    final_score = (
                        similarity_score * 0.6 +
                        skill_score * 0.3 +
                        (experience_years / 10) * 0.1
                    ) * 100

                    clean_result = {
                        "resume_name": filename,
                        "similarity_score": float(similarity_score),
                        "skill_score": float(skill_score),
                        "experience_years": int(experience_years),
                        "skills_found": skills_found,
                        "final_score": float(round(final_score, 2))
                    }

                    results.append(clean_result)
                    results_collection.insert_one(clean_result)

                df = pd.DataFrame(results)
                df = df.sort_values(by="final_score", ascending=False)

                st.subheader("📊 Resume Ranking")
                st.dataframe(df, use_container_width=True)

                # CSV Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Results as CSV",
                    csv,
                    "resume_results.csv",
                    "text/csv"
                )

                # Visualization
                st.subheader("📈 Score Visualization")

                fig, ax = plt.subplots()
                ax.bar(df["resume_name"], df["final_score"])
                ax.set_ylabel("Final Score")
                ax.set_xlabel("Resume")
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.success("Analysis Complete ✅")