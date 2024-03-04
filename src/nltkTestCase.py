import nltk
import os
import sklearn
from docx import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

sentence = """At !!!!! eight o'clock on Thursday morning
... Arthur didn't feel very good."""

tokens = nltk.word_tokenize(sentence)
resumes="Restructured IAM security and server policies to comply with upcoming government regulations and maintain business with 400+ B2B customers. Developed automation code for building 100+ pipelines, Kubernetes clusters, and cloud infrastructure, maximizing 60% efficiency, performance, and security throughout the website Experience"
# tokens = [word for word in tokens if word.isalnum()] 
# stop_words = set(stopwords.words("english"))
tokens=[]
tokens.append("Restructured IAM security and server policies to comply with upcoming government regulations and maintain business with 400+ B2B customers.Developed automation code for building 100+ pipelines, Kubernetes clusters, and cloud infrastructure, maximizing 60% efficiency, performance, and security throughout the websiteExperienceRestructured IAM security and server policies to comply with upcoming government regulations and maintain business with 400+ B2B customers.eveloped automation code for building 100+ pipelines, Kubernetes clusters, and cloud infrastructure, maximizing 60% efficiency, performance, and security throughout the websiteExperience")
documents = " ".join(tokens) + "Testing Resume Text "

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
print(tokens)