
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sota_list import *
from tqdm import tqdm
from utils.data_utils import FineTunedDataset

import numpy as np
import pickle
import torch
import torch.nn as nn

labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

def load_data():

    processed_data, labels = [], []

    with open('fine_tuned_normal.pkl', 'rb') as file:
        data = pickle.load(file)
        processed_data, labels = list(zip(*data))

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

    #metrics = {
    #    f"{id2label[i]} - accuracy": accuracy_score(y_true[:, i], y_pred[:, i]) 
    #    for i in range(len(labels))
    #}

    overall_accuracy = accuracy_score(y_true, y_pred)

    # Store and return as dictionary
    #metrics['accuracy'] = overall_accuracy
    
    return None

def label_accuracy(y_true, y_pred, labels):

    # Element-wise comparison to find exact matches
    matches = torch.eq(y_true, y_pred)

    # Calculate accuracy for each label
    label_accuracies = matches.float().mean(dim=0).tolist()

    return label_accuracies

# Load the processed dataset
# Split using K-Fold cross validation (4)
data, labels = load_data()
skf = KFold(n_splits=4, shuffle=False)
print("Loaded the data! \n")

# Prepare the training parameters
learning_rate = 0.001
batch_size = 32
epochs = 20
print("Hyperparameters Initialized!\n")

# Convert to tensors
data = torch.stack([sample for sample in data])
labels = torch.tensor(labels)

# Split between train and test
for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):

    # Load the model required: LSTM, GRU, CNN
    print(f"Starting Fold: {fold + 1}")

    model = load_model('cnn')

    # Perform the split
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    # Create the dataloader

    # Initialize DataLoader
    train_dataset = FineTunedDataset(train_data, train_labels)
    test_dataset = FineTunedDataset(test_data, test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Loss functions and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(0, epochs):

        for batch in tqdm(train_loader, ncols = 50):

            data, gold_labels = batch

            # Train the model (Train data)
            # Pass in the first batch as testing
            # Get output for each epoch
            pred_labels = model(data)

            # Display the metrics

            # Calculate the loss
            loss = criterion(pred_labels, gold_labels)

            # Update the loss and gradients
            optimizer.zero_grad()
            loss.backward()

            # Update optimizer
            optimizer.step()
        
        # Get the predictions and output (Test data)

        ...

    # [TODO]: Model related
    # - Get the accuracies for each label (average)
    # - Get the overall accuracy
    # - Check if it is higher than the next highest
    # - Store the model in a dictionary
    # - Save the best model