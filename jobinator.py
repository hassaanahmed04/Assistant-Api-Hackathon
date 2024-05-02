import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import spacy
import streamlit as st
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.write("""
# Job Advisor
""")

label = "Job Description"

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation and len(token) > 1]
    return tokens

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def extract_text_from_pdf(cv_pdf):
    text = ""
    pdf_reader = PdfReader(cv_pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_similarity_score(job_description, cv_text):
    preprocessed_job_description = preprocess_text(job_description)
    lemmatized_job_description = lemmatize_tokens(preprocessed_job_description)
    filtered_job_description = remove_stopwords(lemmatized_job_description)
    processed_job_description = ' '.join(filtered_job_description)

    preprocessed_cv_text = preprocess_text(cv_text)
    lemmatized_cv_text = lemmatize_tokens(preprocessed_cv_text)
    filtered_cv_text = remove_stopwords(lemmatized_cv_text)
    processed_cv_text = ' '.join(filtered_cv_text)

    vectorizer = TfidfVectorizer()
    job_description = vectorizer.fit_transform([processed_job_description])
    cv_text = vectorizer.transform([processed_cv_text])

    cosine_sim = cosine_similarity(job_description, cv_text)
    return cosine_sim[0][0]

def extract_keywords(text):
    tokens = preprocess_text(text)
    keywords = lemmatize_tokens(tokens)
    return keywords

job_description = st.text_area(label, value="", height=None, max_chars=2000, key=None, placeholder="Enter Your Job Descripttion Here")

cv_pdf = st.file_uploader('Pick Your Resume....')

submitButton = st.button("Submit", type="primary")

if submitButton:
    cv_text = extract_text_from_pdf(cv_pdf)
    relevancy_score = get_similarity_score(job_description, cv_text)
    st.write("Relevancy Score:", relevancy_score * 100)

    st.write("Common Keywords:")
    job_keywords = extract_keywords(job_description)
    cv_keywords = extract_keywords(cv_text)
    common_keywords = set(job_keywords).intersection(set(cv_keywords))
    # st.write(common_keywords)

    keyword_counts = {}
    for keyword in common_keywords:
        keyword_counts[keyword] = job_keywords.count(keyword) + cv_keywords.count(keyword)

    fig, ax = plt.subplots()
    ax.bar(keyword_counts.keys(), keyword_counts.values())
    plt.xticks(rotation=45)
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Common Keywords')
    total_keywords = len(job_keywords) + len(cv_keywords)
    keywords_text = ' '.join(common_keywords)

    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    # st.pyplot(fig)
    top_n = 10  
    top_keywords = dict(sorted_keywords[:top_n])

    total_keywords = sum(top_keywords.values())

    keyword_percentages = {keyword: (count / total_keywords) * 100 for keyword, count in top_keywords.items()}

    fig2, ax = plt.subplots()
    ax.pie(keyword_percentages.values(), labels=keyword_percentages.keys(), autopct='%1.1f%%')
    ax.set_title('Top {} Common Keywords'.format(top_n))
    st.pyplot(fig2)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)

    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title('Common Keywords Word Cloud')
    ax.axis('off')
    st.pyplot(fig3)
   
    

    
    