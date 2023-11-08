import spacy
from spacy.training import Example
import random
#    ("Check the performance of CAD treasury bonds today", {"entities": [(25, 28, "currency"), (41, 46, "time_interval"), (29,44, "instrument")]}),
#     ("What was the closing value of USD treasury bonds yesterday?", {"entities": [(30, 33, "currency"), (49, 59, "time_interval"), (34,49, "instrument")]}),
#     ("Get me the summary for CAD treasury bonds from six months ago", {"entities": [(18, 21, "currency"), (38, 52, "time_interval"), (22,37, "instrument")]}),
#     ("Provide a breakdown of USD treasury bonds for the last month", {"entities": [(23, 26, "currency"), (42, 52, "time_interval"), (27,42, "instrument")]}),
#     ("I want to review CAD treasury bonds over the past year", {"entities": [(17, 20, "currency"), (44, 52, "time_interval"), (21,36, "instrument")]}),
#     ("USD treasury bond index figures from a year ago, please", {"entities": [(0, 3, "currency"), (39, 47, "time_interval"), (4,24, "instrument")]}),
#     ("Show the trend for CAD treasury bonds from a year ago", {"entities": [(19, 22, "currency"), (42, 50, "time_interval"), (23,38, "instrument")]}),
#     ("Pull up the CAD treasury bonds since today", {"entities": [(12, 15, "currency"), (31, 36, "time_interval"), (16,31, "instrument")]}),
#     ("Give me USD treasury bonds performance from last week", {"entities": [(8, 11, "currency"), (40, 49, "time_interval"), (12,27, "instrument")]})
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
test_sentence = "Provide me CAD bonds from yesterday"
doc = nlp(test_sentence)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
