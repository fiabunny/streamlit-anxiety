import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from bs4 import BeautifulSoup
import string
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the cleaned dataset
df = pd.read_excel('cleaned_data.xlsx') 

# Ensure no NaN values in the 'cleaned' column
df['cleaned'] = df['cleaned'].fillna('')

X = df.cleaned
y = df.label

# Load model and vectorizer
vect = CountVectorizer(max_features=20000, lowercase=False, ngram_range=(1, 2))

# Check for NaN or empty strings in the dataset and fill or remove them
X = X.fillna('')

# Apply CountVectorizer to the cleaned text data
X_cv = vect.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.3, random_state=1, stratify=y)

# Train Decision Tree model
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)

# Save the trained model
filename = 'CV_BestModel.sav'
pickle.dump(clf_dt, open(filename, 'wb'))

# Initialize necessary objects for text processing
lemmatizer = WordNetLemmatizer()

# Define contraction mapping
contraction_mapping = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", 
    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", 
    "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
    "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 
    "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", 
    "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", 
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 
    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
    "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", 
    "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", 
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", 
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

# Text cleaning function
def text_cleaner(text):
    text = text.lower()
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\([^)]*\)', '', text)  # Removing text inside parentheses
    text = re.sub('"','', text)
    text = ' '.join([contraction_mapping.get(t, t) for t in text.split()])
    text = re.sub(r"'s\b", "", text)  # Removing possessive 's'
    text = re.sub("[^a-zA-Z]", " ", text)  # Removing non-alphabetic characters
    text = re.sub('[m]{2,}', 'mm', text)  # Handling repetition like "mm"
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(lemmatized)
    return text

# Streamlit app setup
st.title("Anxiety Detection")

# User input section
user_input = st.text_area("Enter a text to analyze:")

# When user clicks on the analyze button
if st.button("Analyze"):
    if user_input.strip():  # Ensure text is not empty
        # Preprocess the input text
        clean_text = text_cleaner(user_input)
        
        loaded_model = pickle.load(open('CV_BestModel.sav', 'rb'))
        single_prediction = loaded_model.predict(vect.transform([clean_text]).toarray())[0]

        # Map prediction to output
        output = {0: "No Anxiety", 1: "Anxiety"}
        st.write("Prediction:", output[single_prediction])
    else:
        st.write("Please enter some text for analysis.")
