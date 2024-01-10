
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sota_list import *
from utils.data_utils import FineTunedDataset

import numpy as np
import pickle
import torch

labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

def load_data():

    processed_data, labels = [], []

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

def multi_label_metrics(pred_logits, gold_labels):

    # Our threshold
    threshold = 0.5

    # Apply sigmoid to logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred_logits))

    # Convert predictions to integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    # Compute metrics
    y_true = gold_labels

    metrics = {
        f"{id2label[i]} - accuracy": accuracy_score(y_true[:, i], y_pred[:, i]) 
        for i in range(len(labels))
    }

    overall_accuracy = accuracy_score(y_true, y_pred)

    # Store and return as dictionary
    metrics['accuracy'] = overall_accuracy
    
    return metrics

def label_accuracy(y_true, y_pred, labels):

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

# Split between train and test

# Create the dataloader

train_dataset = FineTunedDataset()
test_dataset = FineTunedDataset()

# Train the model

# Get the predictions and output

# Display the metrics

# Save the model


