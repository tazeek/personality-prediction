
from transformers import BertTokenizer, BertModel
from pathlib import Path
#from pymongo import MongoClient, UpdateOne
from keras import backend as K
from unseen_predictor import extract_bert_features

import utils.dataset_processors as dataset_processors
import tensorflow as tf
import json
import codecs
import pickle
import sys
import time

"""
def get_database_connector(collection_name):
    cluster = MongoClient()
    database = cluster['PersonalityFeatures']
    collection =  database[collection_name]

    return collection
"""

def store_database(collection, data):

    # Update the operations list
    operations_list = []

    collection.bulk_write(operations_list)

    return None

def get_personality_features(embeddings, personality_models):
     
     personality_features = {}

     # Iterate trait by trait
     for trait, model in personality_models.items():

        outputs = [
            K.function([model.input],[layer.output])([embeddings])
            for layer in model.layers
        ]

        # Get the features before the softmax
        personality_features[trait] = outputs[0][0][0]

     return personality_features

def load_dataset():
     
    file_directory = "dataset_erc/dailydialogue/dev.json"

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
        models[trait] = tf.keras.models.load_model(model_name)

    return models

def get_personalities_conversation(conversation, tokenizer, bert_model, personality_models):
     
     new_conversation = []

     for dialog in conversation:
          
        # Get the utterance
        utterance = dialog['utterance']
        cleaned_utterance = dataset_processors.preprocess_text(utterance)
        
        # Convert utterance to embeddings
        embeddings = extract_bert_features(
             utterance,
             tokenizer,
             bert_model,
             512,
             256
        )

        # Feed embeddings into the model
        # Get 5 feature vectors
        personality_features_dict = get_personality_features(
             embeddings,
             personality_models
        )

        # [Bonus] Convert features to VAD domain
        # Refer to the paper
        valence_embedding = (personality_features_dict['EXT'] * 0.21) + \
            (personality_features_dict['AGR'] * 0.59) + \
            (personality_features_dict['NEU'] * 0.19)
        
        arousal_embedding = (personality_features_dict['OPN'] * 0.15) + \
            (personality_features_dict['AGR'] * 0.30) - \
            (personality_features_dict['NEU'] * 0.57)
        
        dominance_embedding = (personality_features_dict['OPN'] * 0.25) + \
            (personality_features_dict['CON'] * 0.17) + \
            (personality_features_dict['EXT'] * 0.60) - \
            (personality_features_dict['AGR'] * 0.32)
        
        personality_features_dict['valence'] = valence_embedding
        personality_features_dict['arousal'] = arousal_embedding
        personality_features_dict['dominance'] = dominance_embedding

        # Save features to dictionary
        dialog['personality_features'] = personality_features_dict
        new_conversation.append(dialog)

     return new_conversation


# Load the pre-trained models
# - Big 5
# - BERT
print("Loading BERT models")
bert_tokenizer, bert_model = get_bert_model()

print("Loading OCEAN modes")
models_ocean = load_finetuned_models()

# Load the dataset
print("Loading dataset")
dataset = load_dataset()

personality_conv_dataset = []

# Iterate by conversation
start = time.time()
for index, conversation in enumerate(dataset):
     
    # Iterate by utterance
    personalized_conversation_dict = get_personalities_conversation(
          conversation,
          bert_tokenizer,
          bert_model,
          models_ocean
    )

    # Append to list
    personality_conv_dataset.append(personalized_conversation_dict)

    if (index + 1) % 100 == 0:
          print(f"{index + 1} conversations processed")

time_taken = time.time() - start
print(f"Total time taken: {time_taken:.3f} seconds")

# Save dictionary to either pickle or JSON
# [Bonus] Save every utterance in a MongoDB record
with open('dev_dataset_vad.pkl', 'wb') as file:
    pickle.dump(personality_conv_dataset, file)