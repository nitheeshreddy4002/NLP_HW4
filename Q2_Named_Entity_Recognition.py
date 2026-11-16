
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Input sentence
sentence = " LeBron James as the NBA player of the United States in Laker and won the NBA Final MVP in 2020."

# Process the sentence using spaCy
doc = nlp(sentence)

# Extract and display named entities
print("Named Entities:")

for ent in doc.ents:
    print("Entity:", ent.text)
    print("Label:", ent.label_)
    print("Start Position:", ent.start_char)
    print("End Position:", ent.end_char)
    print("-" * 40)
