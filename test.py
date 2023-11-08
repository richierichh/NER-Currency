import spacy
from spacy.tokens import Doc

# Let's assume 'nlp' is your loaded or blank spaCy language model
nlp = spacy.blank('en')

# Example text and token-based entities
token_based_entities = [
    ('Check the performance of CAD bonds since last month'),
    ('What was the closing value of USD bonds from the previous day?')
]

# Convert token-based entities to character-based entities
char_based_entities = []
for text, annotation in token_based_entities:
    doc = nlp(text)  # Create a Doc object from the text
    entities = []
    for start_token, end_token, label in annotation['entities']:
        start_char = doc[start_token].idx
        # End char is the start of the last token plus its length
        end_char = doc[end_token - 1].idx + len(doc[end_token - 1])
        entities.append((start_char, end_char, label))
    char_based_entities.append((text, {'entities': entities}))

# Now 'char_based_entities' contains character-based indices
print(char_based_entities)
