
from transformers import BertTokenizer, BertModel
from pathlib import Path

from unseen_predictor import extract_bert_features
import utils.dataset_processors as dataset_processors

import tensorflow as tf

import json
import codecs
import torch
import sys

def get_personality_features():
     
     # Iterate trait by trait
     # Get the features before the softmax
     # Save in dictionary
     ...

def load_dataset():
     
    file_directory = 'dataset_erc\dailydialogue\dev.json'

    train_file = []

    with codecs.open(file_directory, "r", "utf-8") as f:
        train_file = json.load(f)

    return train_file

def get_bert_model():

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    return tokenizer, model

def load_finetuned_models():
    directory = 'pkl_data/'
    fine_tuned_name = 'mlp_lm'
    trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    path_model = directory + "finetune_" + str(fine_tuned_name).lower()

    # Check if directory exists
    if not Path(path_model).is_dir():
        print(f"The directory with the selected model was not found: {path_model}")
        sys.exit(0)

    # Load the models and store in dictionary
    models = {}

    for trait in trait_labels:
            
            model_name = f"{path_model}/MLP_LM_{trait}.h5"
            print(f"Load model: {model_name}")
            models[trait] = tf.keras.models.load_model(model_name)

    return models

def get_personalities_conversation(conversation):
     
     for utterance in conversation:
          
        # Get the utterance

        # Convert utterance to embeddings

        # Feed embeddings into the model
        # Get 5 feature vectors

        # [Bonus] Convert features to VAD domain
        # Refer to the paper

        # Save features to dictionary
          ...

# Load the pre-trained models
# - Big 5
# - BERT
bert_tokenizer, bert_model = get_bert_model()
models_ocean = load_finetuned_models()

# Load the dataset
dataset = load_dataset()

# Iterate by conversation
for conversation in dataset:
     
    # Iterate by utterance
     personalized_conversation_dict = get_personalities_conversation(conversation)

     # Append to list

# Save dictionary to either pickle or JSON
# [Bonus] Save every utterance in a MongoDB record