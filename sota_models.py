
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from pprint import pprint
from torch import cuda

from sota_list import *
from tqdm import tqdm
from utils.data_utils import FineTunedDataset

import random
import numpy as np
import pickle
import torch
import torch.nn as nn

person_labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']
id2label = {idx:label for idx,label in enumerate(person_labels)}

def file_name_mapper(file_name):
    return {
        'normal': 'fine_tuned_normal',
        'segmented': 'fine_tuned_sentence_segmentation'
    }[file_name]

def load_data(file_name):

    processed_data, labels = [], []

    # Get the file name
    file_name = file_name_mapper(file_name)
    print(f"Loading file: {file_name}.pkl\n")

    with open(f'{file_name}.pkl', 'rb') as file:
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

    # TODO: 
    metrics_dict = {}
    for i in range(len(person_labels)):

        key = f"{id2label[i]}"

        # Check for F1
        metrics_dict[f"{key} - f1"] = f1_score(gold_labels[:, i], y_pred[:, i])

        # Check for accuracy
        metrics_dict[f"{key} - accuracy"] = accuracy_score(gold_labels[:, i], y_pred[:, i])

    # Perform checking
    #metrics = {
    #    f"{id2label[i]} - accuracy": accuracy_score(gold_labels[:, i], y_pred[:, i]) 
    #    for i in range(len(person_labels))
    #}

    metrics_dict['overall_accuracy'] = accuracy_score(gold_labels, y_pred)
    metrics_dict['overall_f1'] = f1_score(gold_labels, y_pred, average='micro')

    print("\n")
    pprint(metrics_dict)
    print("\n")

    return metrics_dict

# Fix random state
seed = 42
np.random.seed(seed)
random.seed(seed)
cuda.manual_seed(seed)
cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)

# Load the processed dataset (Options: normal, segmented)
file_name = 'segmented'
data, labels = load_data(file_name)

# Split using K-Fold cross validation (4)
folds = 4
skf = KFold(n_splits=folds, shuffle=True, random_state=seed)
print("Loaded the data! \n")

# Prepare the training parameters
learning_rate = 0.001
batch_size = 32
epochs = 20
drop_last = True
model_name = 'lstm'
print("Hyperparameters Initialized!\n")

# Convert to tensors
data = torch.stack([sample for sample in data])
labels = torch.FloatTensor(labels)

# Full dictionary
highest_overall_accuracy = 0
best_model_metrics = {}
best_model_fold = None
best_model = None
full_metrics = {}

# Split between train and test
for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):

    # Load the model required: LSTM, GRU, CNN (WORKS)
    # Create model and mount on GPU
    print(f"Initializing model: {model_name}\n\n")
    model = load_model(model_name)
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

        print(f"Starting Fold Number {fold + 1} (Epoch: {epoch + 1})")
        total_loss = 0.0

        # Train the model (Train data)
        model.train()

        for batch in tqdm(train_loader, ncols = 50):

            sample_data, gold_labels = batch

            # - Mount the data and labels to GPU here
            sample_data = sample_data.cuda()
            gold_labels = gold_labels.cuda()

            # Get output
            pred_labels = model(sample_data)
            
            # Calculate the loss
            loss = criterion(pred_labels, gold_labels)
            total_loss += loss.cpu().item()

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

            sample_data, gold_labels = batch

            # Mount the data and labels to GPU here
            sample_data = sample_data.cuda()

            # Get output
            pred_labels = model(sample_data)

             # Mount the predictions to CPU
            pred_labels = pred_labels.cpu().detach().numpy()
            gold_labels = np.array(gold_labels)

            # Add to the list for metrics checking
            predicted_output.extend(pred_labels)
            gold_labels_list.extend(gold_labels)

        # Display the metrics
        new_metrics = multi_label_metrics(predicted_output, gold_labels_list)
        print(f"Total loss: {total_loss}\n\n")

    # We only take the last metric update
    full_metrics = {
        key: full_metrics.get(key, 0) + new_metrics.get(key, 0) 
        for key in new_metrics.keys()
    }

    # Find the best model via accuracy check
    if full_metrics['overall_accuracy'] > highest_overall_accuracy:
        highest_overall_accuracy = full_metrics['overall_accuracy']
        best_model = model
        best_model_metrics = new_metrics
        best_model_fold = fold
    
    print("=" * 20)
    print("\n")

    del model

# Average them out by the number of folds
full_metrics = {
    key: float(full_metrics.get(key, 0)/folds)
    for key in full_metrics.keys()
}

# Display here
print("OVERALL AFTER AVERAGING\n")
pprint(full_metrics)

print(f"\nBest model metrics is in Fold {best_model_fold + 1}:\n")
pprint(best_model_metrics)

print("\nBest model\n")
print(best_model)

# Save the model
best_model_name = f"finetuned_saved_models/{model_name}_{file_name}.pth"
print(f"Model name is: {best_model_name}")
#torch.save(best_model.state_dict(), best_model_name)

