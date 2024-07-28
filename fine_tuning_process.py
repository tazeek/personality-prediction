from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLNetTokenizer, ElectraModel, ElectraTokenizer, AlbertModel, AlbertTokenizer

import utils.dataset_processors as dataset_processors

import argparse

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
    parser.add_argument("--split", "--sp", type=float, default=0.7)
    parser.add_argument("--dataset", "-ds", type=str, default="essays")
    parser.add_argument("--epoch", "-ep", type=int, default=10)
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

    ...

def transform(tokenizer, dataset):

    ...

def fine_tuning_approach(approach_style, dataset):

    # Whole sentence

    # Sentence
    ...

def start_fine_tuning(model, epochs, train_set, test_set):
    ...


