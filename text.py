import spacy 
from spacy import displacy 
from spacy.training import Example 
import random 

TRAIN_DATA = [
    ("Check the performance of CAD bonds since last month", {"entities": [(25, 28, "currency"), (41, 52, "time_interval")]}),
    ("What was the closing value of USD bonds from the previous day?", {"entities": [(30, 33, "currency"), (49, 65, "time_interval")]}),
    ("Get me last week's summary for CAD bonds", {"entities": [(31, 34, "currency"), (7, 19, "time_interval")]}),
    ("Provide a breakdown of USD bond yields from yesterday", {"entities": [(23, 26, "currency"), (44, 54, "time_interval")]}),
    ("I want to review CAD bond rates over the last night", {"entities": [(17, 20, "currency"), (41, 52, "time_interval")]}),
    ("USD bond index figures for the last week, please", {"entities": [(0, 3, "currency"), (31, 40, "time_interval")]}),
    ("Show the trend for CAD bonds since the fiscal year started", {"entities": [(20, 23, "currency"), (37, 61, "time_interval")]}),
    ("Can you give me the historical prices for USD bonds since Q2 began?", {"entities": [(41, 44, "currency"), (51, 60, "time_interval")]}),
    ("Pull up the daily returns for CAD bonds for the most recent month", {"entities": [(29, 32, "currency"), (53, 70, "time_interval")]}),
    ("Generate a report on USD bonds for the first half of the year", {"entities": [(22, 25, "currency"), (42, 65, "time_interval")]})
]

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

nlp.to_disk(r'C:\Users\Richie\Desktop\models')
sentence = "Show me the AUD bonds from yesterday"
doc = nlp(sentence)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])