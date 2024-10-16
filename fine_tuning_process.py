from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, XLNetForSequenceClassification, XLNetTokenizer, ElectraForSequenceClassification, ElectraTokenizer, AlbertForSequenceClassification, AlbertTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data_utils import DatasetLoader

import utils.dataset_processors as dataset_processors

import pandas as pd
import re
import argparse

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

def _prepare_dataloader(dataset):

    labels_list = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

    text_entries = []
    label_entries = []

    for _, row in dataset.iterrows():

        # Process the text
        text_entries.append(dataset_processors.preprocess_text(row['text']))

        # Process the labels
        label_entries.append([row[label] for label in labels_list])

    # Feed into dataloader and return
    return DatasetLoader(text_entries, label_entries)

def _dataset_directory(name):
    return {
        'essays': "data/essays/essays.csv"
    }[name]

def load_default_hyperparams():

    parser = argparse.ArgumentParser()

    # Model related
    parser.add_argument("--llm_name", "-pm", type=str, default="bert",
        choices=["bert", "roberta", "xlnet", "electra", "albert"])
    
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
        labels = 5,
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

def transform_dataloader(use_sentence_segmentation, dataset):

    new_dataentries_list = []

    # Iterate one at a time
    if use_sentence_segmentation:
        for _, row in dataset.iterrows():

            # Add to the dictionary list
            new_dataentries_list.extend(_sentence_segmentation_process(row))

        # Turn the dictionary list into a dataframe
        dataset = pd.DataFrame(new_dataentries_list)

    # Transformation and return the DataLoader
    return _prepare_dataloader(dataset)

def start_fine_tuning(model, tokenizer, train_set):

    # Initiate optimizer
    model.train()

    optimizer = Adam(model.parameters(), weight_decay=1e-8)
    loss_function = BCEWithLogitsLoss()

    # Start iterating the batch
    for batch_set in tqdm(train_set, n_cols=50):
        input_text, input_labels = batch_set

        # Transform the dataset (Tokenizer)
        input_text = tokenizer(input_text, truncation = True)

        # Feed into the model

        # Get the output
        # Find the loss
        loss = ...

        # Update the model weights and gradients

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ...

# Load hyperparameters settings
args_settings = load_default_hyperparams()

# Load the dataset
dataset_name = "essays"
dataset_full = load_dataset(dataset_name)

# Split the dataset: 60% for Training, 20% for Testing, 20% for validation
train, test, validation = splitting(dataset_full, args_settings.train_split)

# Transform the dataset (DataLoader)
train_set = transform_dataloader(args_settings.sentence_segmentation, train)

train_loader = DataLoader(train_set, args_settings.batch_size, shuffle=False)

# Load the LLMs
model, tokenizer = load_llm_model(args_settings.llm_name)

for epoch in range(args_settings.epoch + 1):

    # Train the model
    start_fine_tuning(model, tokenizer, train_loader)

    # Evaluate on the validation dataset

    # Evaluate on the test dataset

    # Display the update per epoch (Validation + Test)

    quit()

# Save the model

