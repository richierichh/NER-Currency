import spacy
from spacy.training import Example
import random
import time 
import tkinter as tk
from tkinter import scrolledtext
start_time = time.time() 

def generate_sentence(currency, instrument, time_interval_phrase):
    sentence = f"Check the performance of {currency} {instrument} since {time_interval_phrase}"
    return sentence

def find_entity_indices(sentence, phrases):
    return [(sentence.find(phrase), sentence.find(phrase) + len(phrase), label) 
            for phrase, label in phrases]

def add_labels_to_ner(ner, labels):
    for label in labels:
        ner.add_label(label)

def train_ner_model(data, nlp, iterations=100):
    ner = nlp.get_pipe('ner')
    optimizer = nlp.begin_training()
    for itn in range(iterations):
        random.shuffle(data)
        losses = {}
        for text, annotations in data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
        print(losses)

# Initialize spacy model and NER component
nlp = spacy.blank('en')
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)

# Define data structures
instruments = ["bonds", "treasury bonds"]
time_intervals = ["today", "yesterday", "last week", "last month", "six months ago", "a year ago"]
currencies = ["CAD", "USD"]

# Generate training data
TRAIN_DATA = []
entity_labels = set()

for currency in currencies:
    for time_interval in time_intervals:
        for instrument in instruments:
            sentence = generate_sentence(currency, instrument, time_interval)
            phrases = [
                (currency, 'currency'),
                (time_interval, 'time_interval'),
                (instrument, 'instrument')
            ]
            entities = find_entity_indices(sentence, phrases)
            TRAIN_DATA.append((sentence, {"entities": entities}))
            entity_labels.update([label for _, _, label in entities])

# Add entity labels to NER component
add_labels_to_ner(ner, entity_labels)

# Train NER model
with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
    train_ner_model(TRAIN_DATA, nlp)

# Save the model to disk (optional)
# nlp.to_disk(r'C:\Users\Richie\Desktop\models')

# Test the trained model
test_sentence = "Give me EURO treasury bonds from last month"
doc = nlp(test_sentence)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")