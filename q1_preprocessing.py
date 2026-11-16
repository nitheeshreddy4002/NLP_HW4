import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess(sentence):

    # Step 1: Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Step 2: Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))

    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stop_words
        and word not in string.punctuation
    ]

    # Step 3: Apply stemming
    stemmer = PorterStemmer()

    stemmed_words = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    # Print results
    print("Original Tokens:", tokens)
    print("Tokens Without Stopwords:", filtered_tokens)
    print("Stemmed Words:", stemmed_words)


# Input sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

# Run preprocessing
preprocess(sentence)