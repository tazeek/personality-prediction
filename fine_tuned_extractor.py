from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.model_selection import StratifiedKFold
from sota_list import *

import torch
import pickle
import utils.dataset_processors as dataset_processors

# Load BERT tokenizer and model (fine-tuned)
config = AutoConfig.from_pretrained('fine-tuned-bert-personality', output_hidden_states =True)
model = AutoModelForSequenceClassification.from_pretrained('fine-tuned-bert-personality', config=config)
tokenizer = AutoTokenizer.from_pretrained('fine-tuned-bert-personality')

labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

def prepare_data(row):

    # Tokenize sentence
    essays = row['text']

    # Encode them using the tokenizer
    encoded_essay = tokenizer(essays, truncation = True, return_tensors='pt')

    # Convert to embeddings via CLS token
    cls_embedding = []

    with torch.no_grad():
        bert_output = model(**encoded_essay)
        
        # CLS Token extraction
        # - Last layer
        # - Shape (Batch size x Number of tokens x Number of features)
        # - CLS for single sample: 0, 0, all 768
        cls_embedding = bert_output.hidden_states[-1][0,0,:]
        
    # Merge the labels
    merged_labels = [row[key] for key in row.keys() if key in labels]

    # Return the sentence and label
    return cls_embedding, merged_labels

# Load the dataset and pre-process
# Split using K-Fold cross validation (4)
datafile = "data/essays/essays.csv"
dataset = dataset_processors.load_essays_df(datafile)

cls_features = []
merged_labels = []

for index, row in dataset.iterrows():
    bert_output, multi_labels = prepare_data(row)

    cls_features.append(bert_output)
    merged_labels.append(multi_labels)

    if index % 100 == 0:
        print(f"Essays processed: {index + 1}")

with open('fine_tuned_normal.pkl', 'wb') as f:
    pickle.dump(zip(cls_features, merged_labels), f)