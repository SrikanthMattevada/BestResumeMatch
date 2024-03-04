import os
from docx import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to extract text from .doc files
def extract_text_from_doc(docx_file):
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + " "
    return text

# Function for text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and convert to lowercase
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

# Sample resumes directory
resumes_dir = "C:\\Newfolder"

# Extract text from .doc files and preprocess
resumes = []
for filename in os.listdir(resumes_dir):
    if filename.endswith(".docx"):
        docx_file = os.path.join(resumes_dir, filename)
        text = extract_text_from_doc(docx_file)
        preprocessed_text = preprocess_text(text)
        resumes.append(preprocessed_text)

# Job description or profile
job_description = """Azure Devops.
"""

# Preprocess job description
preprocessed_job_description = preprocess_text(job_description)

# Combine job description and resumes for TF-IDF vectorization
documents = [preprocessed_job_description] + resumes

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity between job description and resumes
job_similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

# Find the best match resume
best_match_index = job_similarity_scores.argmax()
best_match_score = job_similarity_scores[best_match_index]
best_match_resume = resumes[best_match_index]

print("Best match resume:")
print(best_match_resume)
print("Similarity score:", best_match_score)
