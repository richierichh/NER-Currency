import spacy 
from spacy.training import Example 
import random 
import time 
from spacy.util import minibatch

# def user_input(prompt): 
#     prompt = input("Please enter your text prompt: ")
#     return prompt 

#Add new entity labels to the NER of the Spacy Model 
# 1. Checks if an NER component exists in the NLP pipelines and if doesn't it will create a new one 
# 2. Iterates over TRAIN_DATA and for each annotation it extracts the entities
# 3. Then for each entity label (currency, time interval and instrument) it is added into the NER model 
def add_entity_labels(nlp, TRAIN_DATA):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2]) #registers each entity as a label so that Spacy can train the model 
    return nlp

def train_ner_model(nlp, TRAIN_DATA, batch_size=4, iterations = 50):
    #with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']): #disables other pipeplines 
        optimizer = nlp.begin_training() # start training the data 
         # Reduce as iterations as low as possible  
        for i in range(iterations): # iterate through the data n times 
            random.shuffle(TRAIN_DATA) #shuffle the data so that the model does not learn the order of the data 
            losses = {} #hashmap to keep track of the losses of the data 
            batches = minibatch(TRAIN_DATA, size=batch_size) #divide data into smaller batch sizes 
            for batch in batches: #iterate through each of the batch sizes 
                texts, annotations = zip(*batch) #seperates the texts and corresponding annotations in the batch 
                docs = [nlp.make_doc(text) for text in texts] #process each text into a doc object which are tokens that spacy can work with 
                examples = [] 
                for i, doc in enumerate(docs): 
                    examples.append(Example.from_dict(doc, annotations[i]))
                nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)
            print(losses)

def main():
   

    TRAIN_DATA = [
    ("Check the performance of CAD bonds since last month", {"entities": [(25, 28, "currency"), (41, 51, "time_interval"), (29,34, "instrument")]}),
    ("What was the closing value of USD bonds from the previous day?", {"entities": [(30, 33, "currency"), (49, 61, "time_interval"), ( 34,39, "instrument")]}),
    ("Get me last week's summary for CAD bonds", {"entities": [(31, 34, "currency"), (7, 18, "time_interval"),( 35,40, "instrument")]}),
    ("Provide a breakdown of USD bonds from yesterday", {"entities": [(23, 26, "currency"), (38, 47, "time_interval"),( 27,32, "instrument") ]}),
    ("I want to review CAD bonds over the last night", {"entities": [(17, 20, "currency"), (36, 46,  "time_interval"),( 21,26, "instrument")]}),
    ("USD bond index figures for the last week, please", {"entities": [(0, 3, "currency"), (31, 41, "time_interval"),( 4,14, "instrument")]}),
    ("Show the trend for CAD bonds from last week", {"entities": [(19, 22, "currency"), (34, 43, "time_interval"),( 23,28, "instrument")]}),
    ("Pull up the CAD bonds for the last month", {"entities": [(12, 15, "currency"), (30, 40, "time_interval"),( 16,21, "instrument")]}),
    ("Give me USD bonds from six months ago", {"entities": [(8, 11, "currency"), (23, 37, "time_interval"),( 12,17, "instrument")]}), 
    ("Give me CAD treasury bonds from a month ago", {"entities": [(8, 11, "currency"), (32, 41, "time_interval"),( 12,26, "instrument")]}), 
    ] 
    sentence = input("Enter your text prompt: ")
    start_time = time.time()
    nlp = spacy.blank('en')
    nlp = add_entity_labels(nlp, TRAIN_DATA)
    train_ner_model(nlp, TRAIN_DATA)

    # Test the model

    #sentence = "give me the CAD government bonds for the last week"
    doc = nlp(sentence)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    end_time = time.time()
    print(f"Time taken: {round(end_time - start_time,2)} seconds")

if __name__ == "__main__":
    main()