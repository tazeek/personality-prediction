from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, XLNetForSequenceClassification, XLNetTokenizer, ElectraForSequenceClassification, ElectraTokenizer, AlbertForSequenceClassification, AlbertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data_utils import DatasetLoader

import utils.dataset_processors as dataset_processors

import pandas as pd
import re
import torch
import argparse

def _collate_padding_efficiency(batch_list):
    
    # Find the maximum length, based on tokenization

    # Find the token number for padding

    # Add the padding to the maximum

    # Return the batch list
    return ...

def _sentence_segmentation_process(row):

    # Split sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", row['text'])

    return [{
        'text': sentence,
        'EXT': row['EXT'],
        'NEU': row['NEU'],
        'AGR': row['AGR'],
        'CON': row['CON'],
        'OPN': row['OPN']
    }
        for sentence in sentences
        if len(sentence) != 0       
    ]

def _prepare_dataloader(dataset, tokenizer):

    labels_list = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

    token_entries = []
    attention_entries = []
    label_entries = []

    for _, row in dataset.iterrows():

        # Tokenize and process
        tokenized_text = tokenizer.encode_plus(
            dataset_processors.preprocess_text(row['text']), 
            padding='max_length', 
            max_length = 512,
            truncation=True,
            return_tensors='pt'
        )

        # Append the parts
        token_entries.append(tokenized_text['input_ids'])
        attention_entries.append(tokenized_text['attention_mask'])

        # Process the labels
        label_entries.append([row[label] for label in labels_list])

    # Feed into dataloader and return
    return DatasetLoader(token_entries, attention_entries, label_entries)

def _dataset_directory(name):
    return {
        'essays': "data/essays/essays.csv"
    }[name]

def load_default_hyperparams():

    parser = argparse.ArgumentParser()

    # Model related
    parser.add_argument("--llm_name", "-pm", type=str, default="bert",
        choices=["bert", "roberta", "xlnet", "electra", "albert", "distilbert"])
    
    # Hyperparameters for fine-tuning
    parser.add_argument("--train_split", "--train_split", type=float, default=0.6)
    parser.add_argument("--test_split", "--test_split", type=float, default=0.5)
    parser.add_argument("--dataset", "-ds", type=str, default="essays")
    parser.add_argument("--epoch", "-ep", type=int, default=10)
    parser.add_argument("--sentence_segmentation", "-ss", type=bool, default=True)
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", "--wc", type=float, default=0.01)
    
    return parser.parse_args()


def load_llm_model(model_name):

    # Models used: BERT, RoBERTa, XLNet, ELECTRA, Albert
    model_list = {
        'bert': (BertForSequenceClassification, BertTokenizer, "bert-base-uncased"),
        'roberta': (RobertaForSequenceClassification, RobertaTokenizer, "roberta-base"),
        'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
        'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-uncased'),
        'electra': (ElectraForSequenceClassification, ElectraTokenizer, 'google/electra-base-discriminator'),
        'albert': (AlbertForSequenceClassification, AlbertTokenizer, "albert-base-v2")
    } 

    model_class, tokenizer_class, model_version = model_list[model_name]

    tokenizer = tokenizer_class.from_pretrained(
        model_version,
        do_lower_case = True
    )

    model = model_class.from_pretrained(
        model_version,
        num_labels = 5,
        problem_type = "multi_label_classification",
        output_hidden_states = True
    )

    return model, tokenizer

def load_dataset(dataset_name):

    # Load the dataset
    file_location = _dataset_directory(dataset_name)
    dataset = dataset_processors.load_essays_df(file_location, False)

    return dataset

def splitting(dataset, ratio_split):

    train_data, test_data = train_test_split(dataset, train_size=ratio_split, random_state=42)
    validation_data, test_data = train_test_split(test_data, train_size=0.5, random_state=42)
    
    return train_data, test_data, validation_data

def label_dictionaries(columns):

    labels = [label for label in columns if label not in ['user','text','token_len']]

    # Forward and backward mapping
    id2label = {idx:label for idx,label in enumerate(labels)}
    label2id = {label:idx for idx,label in enumerate(labels)}

    return id2label, label2id

def transform_dataloader(use_sentence_segmentation, dataset, tokenizer):

    new_dataentries_list = []

    # Iterate one at a time
    if use_sentence_segmentation:
        for _, row in dataset.iterrows():

            # Add to the dictionary list
            new_dataentries_list.extend(_sentence_segmentation_process(row))

        # Turn the dictionary list into a dataframe
        dataset = pd.DataFrame(new_dataentries_list)

    # Transformation and return the DataLoader
    return _prepare_dataloader(dataset, tokenizer)

def start_fine_tuning(model, train_set, device):

    # Initiate optimizer
    model.train()

    optimizer = Adam(model.parameters(), weight_decay=0.01, lr=2e-5)
    loss_function = BCEWithLogitsLoss()
    total_loss = 0

    exit_steps = 15

    # Start iterating the batch
    for i, batch_set in enumerate(tqdm(train_set, ncols=50)):

        input_tokens, attention, labels = batch_set

        # Flatten the dimension
        input_tokens = input_tokens.squeeze(1)
        attention = attention.squeeze(1)

        input_tokens = input_tokens.to(device, non_blocking=True)
        attention = attention.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Feed into the model
        outputs = model(
            input_ids = input_tokens,
            attention_mask = attention
        )
        
        # Find the loss
        loss = loss_function(outputs.logits, labels)
        loss.backward()

        total_loss += loss.cpu().item()

        # Update the model weights and gradients
        optimizer.zero_grad()
        optimizer.step()

        if i == exit_steps:
            break
    
    return total_loss

def evaluate_model(model, val_set, device):

    gold_labels = []
    predicted_labels = []

    # Iterate the test set
    for batch_set in tqdm(val_set, ncols=50):

        input_tokens, attention, labels = batch_set

        # Flatten the dimension
        input_tokens = input_tokens.squeeze(1)
        attention = attention.squeeze(1)

        input_tokens = input_tokens.to(device, non_blocking=True)
        attention = attention.to(device, non_blocking=True)

        # Feed into the model
        outputs = model(
            input_ids = input_tokens,
            attention_mask = attention
        )

        # Get the logits and convert to scores

        # Extend and keep with gold labels and predicted labels

    # Evaluate gold label scores

    return None

# Load hyperparameters settings
args_settings = load_default_hyperparams()

# Get CUDA device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device is: {device}")

# Load the LLMs and mount onto CUDA
model, tokenizer = load_llm_model(args_settings.llm_name)
model = model.to(device)

# Load the dataset
dataset_name = "essays"
dataset_full = load_dataset(dataset_name)

# Split the dataset: 60% for Training, 20% for Testing, 20% for validation
train, test, validation = splitting(dataset_full, args_settings.train_split)

# Transform the dataset (DataLoader)
train_set = transform_dataloader(args_settings.sentence_segmentation, train, tokenizer)
test_set = transform_dataloader(args_settings.sentence_segmentation, test, tokenizer)
val_set = transform_dataloader(args_settings.sentence_segmentation, validation, tokenizer)

train_loader = DataLoader(train_set, args_settings.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, args_settings.batch_size, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_set, args_settings.batch_size, shuffle=False, pin_memory=True)

for epoch in range(args_settings.epoch + 1):

    # Train the model
    loss_amount = start_fine_tuning(model, train_loader, device)
    print(loss_amount)

    # Evaluate on the validation dataset

    # Evaluate on the test dataset

    # Display the update per epoch (Validation + Test)

    quit()

# Save the model

