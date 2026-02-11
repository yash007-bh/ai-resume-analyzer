import re
import nltk
import PyPDF2
import docx2txt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom Skill List
SKILLS = [
    "python", "java", "c++", "machine learning",
    "deep learning", "sql", "mongodb", "aws",
    "flask", "django", "spring", "react",
    "tensorflow", "pytorch", "html", "css"
]

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        return docx2txt.process(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_embedding(text):
    return model.encode(text)

def calculate_similarity(resume_emb, jd_emb):
    score = cosine_similarity([resume_emb], [jd_emb])
    return round(score[0][0] * 100, 2)

def extract_skills(text):
    found = []
    text = text.lower()
    for skill in SKILLS:
        if skill in text:
            found.append(skill)
    return list(set(found))

def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())
    years = [int(match[0]) for match in matches]
    return max(years) if years else 0