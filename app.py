import streamlit as st
import pandas as pd
from utils import *
from auth import register_user, login_user
from database import results_collection

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {background-color: #0e1117;}
.stButton>button {background: linear-gradient(90deg,#00C9FF,#92FE9D);}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Resume Screening System 2026")

# Session state login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    menu = st.selectbox("Login / Register", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registered Successfully!")
            else:
                st.error("User already exists")

    if menu == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.success("Login Successful")
            else:
                st.error("Invalid Credentials")

# ---------------- MAIN APP ----------------
else:
 
 
    st.sidebar.success("Logged In")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    jd_text = st.text_area("📄 Paste Job Description")

    uploaded_files = st.file_uploader(
        "📂 Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if st.button("Analyze Resumes"):
        with st.spinner("Analyzing resumes with AI..."):

         
         jd_clean = clean_text(jd_text)
         jd_embedding = get_embedding(jd_clean)

        results = []

        for file in uploaded_files:

            text = extract_text(file)
            cleaned = clean_text(text)

            resume_emb = get_embedding(cleaned)

            similarity = calculate_similarity(resume_emb, jd_embedding)
            skills = extract_skills(cleaned)
            experience = extract_experience(cleaned)

            # Weighted Final Score
            final_score = similarity + (len(skills) * 2) + (experience * 1.5)

            results.append({
                "Resume": file.name,
                "Similarity Score": similarity,
                "Skills Found": ", ".join(skills),
                "Experience (Years)": experience,
                "Final Score": round(final_score, 2)
            })

            # Save to MongoDB
            results_collection.insert_one(results[-1])

        df = pd.DataFrame(results)
        df = df.sort_values(by="Final Score", ascending=False)

        st.dataframe(df, use_container_width=True)
        import matplotlib.pyplot as plt

        st.subheader("Score Visualization")

        fig, ax = plt.subplots()
        ax.bar(df["Resume"], df["Final Score"])
        ax.set_ylabel("Final Score")
        ax.set_xlabel("Resume")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # CSV Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            "resume_results.csv",
            "text/csv"
        )