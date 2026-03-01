import streamlit as st
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the model and scaler
model1 = joblib.load('linear_model.pkl')
model2 = joblib.load('ridge_model.pkl')
model3 = joblib.load('lasso_model.pkl')
scaler = joblib.load('scaler.pkl')


st.title(" 🏠 House Price Prediction ")
st.write("Compare predictions using a suitable model.")

# User inputs
MedInc = st.number_input('Median Income (in 10k$)', min_value=0.0)
HouseAge = st.number_input('House Age', min_value=1.0)
AveRooms = st.number_input('Average Rooms', min_value=1.0)
AveBedrms = st.number_input('Average Bedrooms', min_value=0.0)
Population = st.number_input('Population', min_value=1.0)
AveOccup = st.number_input('Average Occupancy', min_value=1.0)
Latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0)
Longitude = st.number_input('Longitude', min_value=-124.0, max_value=-114.0)

# Prepare input
features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
features = scaler.transform(features)

# Prediction
if st.button('Predict House Value'):
    pred_model1 = model1.predict(features)[0]
    pred_model2 = model2.predict(features)[0]
    pred_model3 = model3.predict(features)[0]

    st.success(f"Linear Regression Model Prediction: ${pred_model1 * 100000:.2f}")
    st.success(f"Ridge Regression Model Prediction: ${pred_model2 * 100000:.2f}")
    st.success(f"Lasso Regression Model Prediction: ${pred_model3 * 100000:.2f}")

# Background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
            url("https://bambooandtimber.com.au/wp-content/uploads/2023/06/Property-Value.jpg");
             background-attachment: fixed;
             background-size: cover;
             backdrop-filter: blur(5px);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    

# Call the function to add background
add_bg_from_url()


import streamlit as st
import pickle
import joblib  # Added this import
import math
from collections import Counter
import nltk


# Load the corpus
with open('corpus.pkl', 'rb') as f:
    Corpus = pickle.load(f)


# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('SimpleTfidfVectorizer.pkl')  # Use joblib, not pickle



# NLTK imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# NLTK setup (downloads only first run)
# -------------------------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Normalize and preprocess user input
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)



# Cosine similarity function
from sklearn.metrics.pairwise import cosine_similarity  # Needed for similarity

def get_response(user_input):
    processed_input = preprocess_text(user_input)
    input_vector = tfidf_vectorizer.transform([processed_input])  # remove [0], cosine_similarity expects 2D

    # Correct preprocessing of patterns (do NOT join characters)
    patterns = [preprocess_text(p) for p, r in Corpus]
    pattern_vectors = tfidf_vectorizer.transform(patterns)

    # Compute cosine similarity
    similarities = cosine_similarity(input_vector, pattern_vectors)
    best_match_idx = similarities.argmax()
    
    # Optional threshold
    if similarities[0][best_match_idx] < 0.2:
        return "Sorry, I don't understand."

    return Corpus[best_match_idx][1]  # Return the response



# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🏠 California Housing Chatbot 🤖")
st.write("Ask me about dataset, preprocessing, or regression models (Linear, Lasso, Ridge).")

user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.text_area("Chatbot:", response, height=100)

# Optional background style
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)),
                    url('https://bambooandtimber.com.au/wp-content/uploads/2023/06/Property-Value.jpg');
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
