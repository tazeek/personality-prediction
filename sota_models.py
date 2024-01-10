
from sklearn.model_selection import StratifiedKFold
from sota_list import *

import pickle
import torch

# Load BERT tokenizer and model (fine-tuned)
config = AutoConfig.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', output_hidden_states =True)
model = AutoModelForSequenceClassification.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', config=config)
tokenizer = AutoTokenizer.from_pretrained('fine-tuned-bert-personality-sentence-segmentation')

labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

def load_data():

    processed_data, labels = []

    with open('fine_tuned_sentence_segmentation.pkl', 'rb') as file:
        data = pickle.load(file)
        processed_data, labels = list(zip(*data))

    print(len(processed_data))
    print(labels)
    quit()
    return processed_data, labels

def load_model(model_name):
    return {
        'cnn': CNN(5),
        'lstm': LSTMNetwork(768,128,5),
        'gru': GRUNetwork(768,128,5)
    }[model_name]

def label_accuracy(y_true, y_pred):

    # Element-wise comparison to find exact matches
    matches = torch.eq(y_true, y_pred)

    # Calculate accuracy for each label
    label_accuracies = matches.float().mean(dim=0).tolist()

    return label_accuracies

# Load the processed dataset
# Split using K-Fold cross validation (4)
data, labels = load_data()
skf = StratifiedKFold(n_splits=4, shuffle=False)

# Prepare the training parameters
learning_rate = 0.001
batch_size = 32
epochs = 20

# Load the model required: LSTM, GRU, CNN
model = load_model('cnn')

# Train the model

# Display the metrics

# Save the model