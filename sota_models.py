
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from pprint import pprint

from sota_list import *
from tqdm import tqdm
from utils.data_utils import FineTunedDataset

import numpy as np
import pickle
import torch
import torch.nn as nn

person_labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']
id2label = {idx:label for idx,label in enumerate(person_labels)}

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

def multi_label_metrics(probs, gold_labels):

    # Our threshold
    threshold = 0.5

    # Convert predictions to integer predictions
    probs = np.array(probs)
    gold_labels = np.array(gold_labels)

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    # Perform checking
    metrics = {
        f"{id2label[i]} - accuracy": accuracy_score(gold_labels[:, i], y_pred[:, i]) 
        for i in range(len(person_labels))
    }

    metrics['overall_accuracy'] = accuracy_score(gold_labels, y_pred)
    print("\n")
    pprint(metrics)
    print("\n")
    return metrics

def label_accuracy(y_true, y_pred):

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
drop_last = True
print("Hyperparameters Initialized!\n")

# Convert to tensors
data = torch.stack([sample for sample in data])
labels = torch.FloatTensor(labels)

# Split between train and test
for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):

    # Load the model required: LSTM, GRU, CNN (WORKS)
    # Create model and mount on GPU
    model = load_model('lstm')
    model.cuda()

    # Perform the split
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    # Initialize DataLoader
    train_dataset = FineTunedDataset(train_data, train_labels)
    test_dataset = FineTunedDataset(test_data, test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    # Loss functions and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(0, epochs):

        print(f"Starting Fold Number {fold + 1} (Epoch: {epoch})")

        # Train the model (Train data)
        model.train()

        for batch in tqdm(train_loader, ncols = 50):

            data, gold_labels = batch

            # Get output
            pred_labels = model(data)
            
            # Calculate the loss
            loss = criterion(pred_labels, gold_labels)

            # Update the loss and gradients
            optimizer.zero_grad()
            loss.backward()

            # Update optimizer
            optimizer.step()
        
        # Get the predictions and output (Test data)
        model.eval()
        predicted_output = []
        gold_labels_list = []

        for batch in tqdm(test_loader, ncols=50):

            data, gold_labels = batch

            # Get output
            pred_labels = model(data)

             # Mount to CPU
            pred_labels = pred_labels.cpu().detach().numpy()
            gold_labels = gold_labels.cpu().detach().numpy()

            # Add to the list for metrics checking
            predicted_output.extend(pred_labels)
            gold_labels_list.extend(gold_labels)

        # Display the metrics
        metrics = multi_label_metrics(predicted_output, gold_labels_list)

    
    print("=" * 20)
    print("\n")