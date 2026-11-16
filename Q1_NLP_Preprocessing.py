
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Input sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

# Step 1: Tokenize the sentence into individual words and punctuation
tokens = word_tokenize(sentence)

# Step 2: Remove common English stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]

# Step 3: Apply stemming to the remaining tokens
stemmer = PorterStemmer()
stemmed_words = [
    stemmer.stem(word) for word in filtered_tokens
]

# Display all three results
print("Original Tokens:", tokens)
print("Tokens Without Stopwords:", filtered_tokens)
print("Stemmed Words:", stemmed_words)
