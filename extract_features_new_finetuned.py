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
    print("\n\n")

    for dialog in conversation:

        # Get the utterance
        utterance = dialog['utterance']
        print(utterance)

        # Perform tokenization
        encoded_utterance = tokenizer(
            utterance,
            truncation = True,
            return_tensors = 'pt'
        )

        with torch.no_grad():
            
            # Get the BERT output
            bert_output = bert_model(**encoded_utterance)

            # Extract the CLS token
            cls_embedding = bert_output.hidden_states[-1][0,0,:]
            cls_embedding = cls_embedding.unsqueeze(0)

            # Push it to the model
            features = perso_model.features_extraction(cls_embedding)

            # Append the output
            new_conversation.extend(features.tolist())
    
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
persona_model.eval()

# Load the DailyDialog dataset
print("Loading dataset")
portion_set = "train"
dataset = load_dataset(portion_set)

# Begin the transformation
personality_dataset = []

for index, conversation in enumerate(dataset):

    # Iterate by utterance
    personality_list = perform_extraction(
        conversation,
        tokenizer,
        model,
        persona_model
    )

    print(len(personality_list))

    quit()

    # Append to list
    personality_dataset.append(personality_list)

    if (index + 1) % 100 == 0:
        print(f"{index+1} conversations processed")

# Save dictionary to either pickle or JSON
# [Bonus] Save every utterance in a MongoDB record
file_name = f"{portion_set}_{model_type}_{train_type}.pkl"

with open(file_name, 'wb') as file:
    pickle.dump(personality_dataset, file)
