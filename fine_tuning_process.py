from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLNetTokenizer, ElectraModel, ElectraTokenizer, AlbertModel, AlbertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

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

def _dataset_directory(name):
    return {
        'essays': "data/essays/essays.csv"
    }[name]

def load_default_hyperparams():

    parser = argparse.ArgumentParser()

    # Model related
    parser.add_argument("--pretrained_model", "-pm", type=str, default="bert",
        choices=["bert", "roberta", "xlnet", "electra", "albert"])
    
    # Hyperparameters for fine-tuning
    parser.add_argument("--train_split", "--tp", type=float, default=0.6)
    parser.add_argument("--test_split", "--tp", type=float, default=0.2)
    parser.add_argument("--dataset", "-ds", type=str, default="essays")
    parser.add_argument("--epoch", "-ep", type=int, default=10)
    parser.add_argument("--sentence_segmentation", "-ss", type=bool, default=False)
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    
    return parser.parse_args()


def load_llm_model(model_name):

    # Models used: BERT, RoBERTa, XLNet, ELECTRA, Albert
    model_list = {
        'bert': (BertModel, BertTokenizer, "bert-base-uncased"),
        'roberta': (RobertaModel, RobertaTokenizer, "roberta-base"),
        'xlnet': (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
        'electra': (ElectraModel, ElectraTokenizer, 'google/electra-base-discriminator'),
        'albert': (AlbertModel, AlbertTokenizer, "albert-base-v2")
    } 

    model_class, tokenizer_class, model_version = model_list[model_name]

    tokenizer = tokenizer_class.from_pretrained(
        model_version,
        do_lower_case = True
    )

    model = model_class.from_pretrained(
        model_version,
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
            print(new_dataentries_list)
            quit()

        # Turn the dictionary list into a dataframe
        dataset = pd.DataFrame(new_dataentries_list)

    # Transformation and return the DataLoader

    quit()

    return None

def start_fine_tuning(model, epochs, train_set, test_set):

    # Begin epoch

    # Start iterating the batch

    ## Transform the dataset (Tokenizer)

    # Feed into the model

    # Get the output

    # Find the loss

    # Update the model weights and gradients

    # Evaluate on the validation dataset

    # Evaluate on the test dataset

    # Display the update per epoch (Validation + Test)

    # Save the model
    ...

## Load the dataset
dataset_name = "essays"
dataset_full = load_dataset(dataset_name)

## Split the dataset: 60% for Training, 20% for Testing, 20% for validation
ratio_split = 0.6
train, test, validation = splitting(dataset_full, ratio_split)

## Transform the dataset (DataLoader)
use_sentence_segmentation = True
train_loader = transform_dataloader(use_sentence_segmentation, train)
print(train_loader)
print(len(train_loader))

## Create the hyperparameters

## Load the LLMs

## Train the model


