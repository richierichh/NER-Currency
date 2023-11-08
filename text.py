import spacy 
from spacy import displacy 
from spacy.training import Example 
import random 

TRAIN_DATA = [
    ("Check the performance of CAD bonds since last month", {"entities": [(25, 28, "currency"), (41, 51, "time_interval"), (29,34, "instrument")]}),
    ("What was the closing value of USD bonds from the previous day?", {"entities": [(30, 33, "currency"), (49, 61, "time_interval"), ( 34,39, "instrument")]}),
    ("Get me last week's summary for CAD bonds", {"entities": [(31, 34, "currency"), (7, 18, "time_interval"),( 35,40, "instrument")]}),
    ("Provide a breakdown of USD bonds from yesterday", {"entities": [(23, 26, "currency"), (38, 47, "time_interval"),( 27,32, "instrument") ]}),
    ("I want to review CAD bonds over the last night", {"entities": [(17, 20, "currency"), (36, 46,  "time_interval"),( 21,26, "instrument")]}),
    ("USD bond index figures for the last week, please", {"entities": [(0, 3, "currency"), (31, 41, "time_interval"),( 4,14, "instrument")]}),
    ("Show the trend for CAD bonds from last week", {"entities": [(19, 22, "currency"), (34, 43, "time_interval"),( 23,28, "instrument")]}),
    ("Pull up the CAD bonds for the last month", {"entities": [(12, 15, "currency"), (30, 40, "time_interval"),( 16,21, "instrument")]}),
    ("Give me USD bonds from six months ago", {"entities": [(8, 11, "currency"), (23, 37, "time_interval"),( 12,17, "instrument")]})
] 
# instruments = ["bonds", "treasury bonds"]

# time_intervals = [
#     ("today", "today"),
#     ("yesterday", "yesterday"),
#     ("last week", "last week"),
#     ("last month", "last month"),
#     ("six months ago", "six months ago"),
#     ("a year ago", "a year ago")
# ]
# currencies = ["CAD", "USD"]

# TRAIN_DATA = []

# for currency in currencies:
#     for time_interval_phrase, time_interval_label in time_intervals:
#         for instrument in instruments:
#             # Construct the sentence
#             sentence = f"Check the performance of {currency} {instrument} since {time_interval_phrase}"
#             # Find the start and end indices for entities
#             currency_idx = sentence.find(currency)
#             time_interval_idx = sentence.find(time_interval_phrase)
#             instrument_idx = sentence.find(instrument)

#             # Append to TRAIN_DATA with the correct format
#             TRAIN_DATA.append((sentence, {"entities": [
#                 (currency_idx, currency_idx + len(currency), "currency"),
#                 (time_interval_idx, time_interval_idx + len(time_interval_phrase), "time_interval"),
#                 (instrument_idx, instrument_idx + len(instrument), "instrument")
#             ]}))

nlp = spacy.blank('en')

# Create a new NER component and add it to the pipeline
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

# Add the new label to the NER component
for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2]) 

# Disable other pipeline components during training
with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
    optimizer = nlp.begin_training()
    for itn in range(100):  # Number of training iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
        print(losses)

#nlp.to_disk(r'C:\Users\Richie\Desktop\models')
sentence = "Give me CAD treasury bonds from a month ago"
doc = nlp(sentence)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])