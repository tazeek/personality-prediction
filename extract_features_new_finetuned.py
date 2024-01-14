from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sota_list import *

import torch
import json
import codecs
import pickle

# TODO:
# - Go by utterance
# - Save in pickle format (file name: {file_type}_{model}_{finetuned})

def load_model(model_name):
    return {
        'cnn': CNN(5),
        'lstm': LSTMNetwork(768,128,5),
        'gru': GRUNetwork(768,128,5)
    }[model_name]

def load_finetuned_bert(model_name):
    config = AutoConfig.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', output_hidden_states =True)
    model = AutoModelForSequenceClassification.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', config=config)
    tokenizer = AutoTokenizer.from_pretrained('fine-tuned-bert-personality-sentence-segmentation')

    return model, tokenizer


def load_pretrained_model(model_type, train_type):
    file_name = f"finetuned_saved_models/{model_type}_{train_type}.pth"

    model = load_model(model_type)
    model.load_state_dict(torch.load(f"{file_name}"))

    return model

def load_dataset(portion_set):
     
    file_directory = f"dataset_erc/dailydialogue/{portion_set}.json"

    train_file = []

    with codecs.open(file_directory, "r", "utf-8") as f:
        train_file = json.load(f)

    return train_file

def perform_extraction(conversation, tokenizer, bert_model, perso_model):

    new_conversation = []
    
    for dialog in conversation:

        # Get the utterance
        utterance = dialog['utterance']
        
        # Perform tokenization

        # Get the BERT output

        # Extract the CLS token

        # Push it to the model

        # Append the output
    
    return new_conversation

# Load BERT tokenizer and model (fine-tuned)
print("Loading BERT model and tokenizer")
bert_model_name = 'fine-tuned-bert-personality-sentence-segmentation'
model, tokenizer = load_finetuned_bert(bert_model_name)

# Load the pre-trained model
print("Loading model")
model_type = 'lstm'
train_type = 'segmented'
persona_model = load_pretrained_model(model_type, train_type)

# Load the DailyDialog dataset
print("Loading dataset")
portion_set = "train"
dataset = load_dataset(portion_set)

# Begin the transformation
personality_dataset = []

for index, conversation in enumerate(dataset):

    # Iterate by utterance
    personalized_conversation_dict = perform_extraction(
        conversation,
        tokenizer,
        model,
        persona_model
    )

    quit()

    # Append to list
    personality_dataset.append(personalized_conversation_dict)

    if (index + 1) % 100 == 0:
        print(f"{index+1} conversations processed")

# Save dictionary to either pickle or JSON
# [Bonus] Save every utterance in a MongoDB record
file_name = f"{portion_set}_{model_type}_{train_type}.pkl"

with open(file_name, 'wb') as file:
    pickle.dump(personality_conv_dataset, file)
