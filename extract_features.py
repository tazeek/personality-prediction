
from transformers import BertTokenizer, BertModel
from pathlib import Path

from unseen_predictor import extract_bert_features
import tensorflow as tf

import torch
import sys

import utils.dataset_processors as dataset_processors

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

# Load the pre-trained models
# - Big 5
# - BERT
bert_tokenizer, bert_model = get_bert_model()
models_ocean = load_finetuned_models()

# Load the dataset

# Iterate by conversation

# Iterate by utterance

# Convert utterance to embeddings

# Feed embeddings into the model

# Get the features before the softmax

# [Bonus] Convert features to VAD domain
# Refer to the paper

# Save features to dictionary

# Save dictionary to either pickle or JSON