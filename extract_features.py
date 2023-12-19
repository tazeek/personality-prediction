
from transformers import BertTokenizer, BertModel

def get_bert_model():

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    return tokenizer, model
# Load the pre-trained models
# - Big 5
# - BERT

bert_tokenizer, bert_model = get_bert_model()

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