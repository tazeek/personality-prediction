from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from evaluator import Evaluator
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

def _get_file_name(llm_model, is_segmented):

    # File name: fine-tuned-{llm} or fine-tuned-bert-segmented
    default = f"fine-tuned-{llm_model}"

    if is_segmented:
        return f"{default}-segmented"
    
    return default

def _get_labels_list():
    return ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

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

    labels_list = _get_labels_list()

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
        'bert': "bert-base-uncased",
        'roberta': "roberta-base",
        'xlnet': 'xlnet-base-cased',
        'distilbert': 'distilbert-base-uncased',
        'electra': 'google/electra-base-discriminator',
        'albert': "albert-base-v2"
    } 

    model_version = model_list[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_version,
        do_lower_case = True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_version,
        num_labels = 5,
        problem_type = "multi_label_classification"
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

    model.train()

    # Initiate optimizer
    optimizer = Adam(model.parameters(), weight_decay=0.01, lr=2e-5)
    total_loss = 0

    # Start iterating the batch
    for i, batch_set in enumerate(tqdm(train_set, ncols=50)):

        optimizer.zero_grad()

        input_tokens, attention, labels = batch_set

        # Flatten the dimension
        input_tokens = input_tokens.squeeze(1)
        attention = attention.squeeze(1)

        input_tokens = input_tokens.to(device)
        attention = attention.to(device)
        labels = labels.to(device)

        # Feed into the model
        outputs = model(
            input_ids = input_tokens,
            attention_mask = attention,
            labels = labels
        )
        
        # Find the loss
        loss = outputs.loss
        total_loss += loss.cpu().item()

        # Update the model weights and gradients
        loss.backward()
        optimizer.step()
    
    return model, total_loss

def evaluate_model(model, val_set, evaluator, device):

    model.eval()

    gold_labels = []
    predicted_labels = []

    # Iterate the test set
    with torch.no_grad():
        
        for batch_set in tqdm(val_set, ncols=50):

            input_tokens, attention, labels = batch_set

            # Flatten the dimension
            input_tokens = input_tokens.squeeze(1)
            attention = attention.squeeze(1)

            input_tokens = input_tokens.to(device)
            attention = attention.to(device)

            # Feed into the model
            outputs = model(
                input_ids = input_tokens,
                attention_mask = attention
            )

            # Extend and keep with gold labels and predicted labels
            logits = outputs.logits.cpu()
            pred_labels = evaluator.convert_predictions(logits)

            predicted_labels.extend(pred_labels)
            gold_labels.extend(labels.numpy())

    # Evaluate gold label scores and display
    evaluator.calculate_scores(gold_labels, predicted_labels)

    return None

# Load hyperparameters settings
args_settings = load_default_hyperparams()
file_name = _get_file_name(args_settings.llm_name, args_settings.sentence_segmentation)
labels_list = _get_labels_list()

evaluator = Evaluator(file_name, labels_list)

# Get CUDA device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device is: {device}")

# Load the LLMs and mount onto CUDA
model, tokenizer = load_llm_model(args_settings.llm_name)
print(model)
quit()
model = model.to(device)

# Load the dataset
dataset_name = "essays"
dataset_full = load_dataset(dataset_name)

# Split the dataset: 60% for Training, 20% for Testing, 20% for validation
train, test, _ = splitting(dataset_full, args_settings.train_split)

# Transform the dataset (DataLoader)
train_set = transform_dataloader(args_settings.sentence_segmentation, train, tokenizer)
test_set = transform_dataloader(args_settings.sentence_segmentation, test, tokenizer)

train_loader = DataLoader(train_set, args_settings.batch_size, shuffle=False)
test_loader = DataLoader(test_set, args_settings.batch_size, shuffle=False)

for epoch in range(args_settings.epoch + 1):

    # Train the model
    print(f"Epoch {epoch}: Training")
    model, loss_amount = start_fine_tuning(model, train_loader, device)
    print(f"Loss: {loss_amount}")

    # Evaluate on the test dataset
    print(f"Epoch {epoch}: Testing")
    evaluate_model(model, test_loader, evaluator, device)

# Save the model and tokenizer
model.save_pretrained('./fine-tuned-sentence-bert')
tokenizer.save_pretrained('./fine-tuned-sentence-bert')

