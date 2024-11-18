from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sota_list import *

import torch
import argparse
import pickle
import utils.dataset_processors as dataset_processors

def get_model_names(model_name):

    return f'{model_name}-finetuned-segmented'

def load_args():

    # Create model name
    parser = argparse.ArgumentParser()

    # Model related
    parser.add_argument("--model_name", "-mn", type=str)

    # Add flags for sliding window related -> TODO:

    return parser.parse_args()

def load_llm_parts(model_name):

    # Load the config, model, and tokenizer
    config = AutoConfig.from_pretrained(model_name, output_hidden_states =True)

    return [
        AutoModelForSequenceClassification.from_pretrained(model_name, config=config),
        AutoTokenizer.from_pretrained(model_name)
    ]

def prepare_data(row, tokenizer):

    # Tokenize sentence
    essay = row['text']

    # Encode them using the tokenizer
    encoded_essay = tokenizer(essay, truncation = True, return_tensors='pt')

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
    return cls_embedding, essay, merged_labels

# Load BERT tokenizer and model (fine-tuned)
args_settings = load_args()
model_name = get_model_names(args_settings.model_name)
file_name = f'{model_name}-extracted'

model, tokenizer = load_llm_parts(model_name)

labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

# Load the dataset and pre-process
datafile = "data/essays/essays.csv"
dataset = dataset_processors.load_essays_df(datafile)

# Create sliding window initializer -> TODO:

cls_features = []
input_samples = []
merged_labels = []

for index, row in dataset.iterrows():

    # Add the different segmentation methods for sliding window -> TODO:

    bert_output, input_sample, multi_labels = prepare_data(row, tokenizer, labels)

    cls_features.append(bert_output)
    input_samples.append(input_sample)
    merged_labels.append(multi_labels)

    if index % 100 == 0:
        print(f"Essays processed: {index + 1}")

with open(f'{file_name}.pkl', 'wb') as f:
    pickle.dump(zip(cls_features, input_samples, merged_labels), f)
