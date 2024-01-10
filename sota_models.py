from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.model_selection import StratifiedKFold
from sota_list import *

import torch
import utils.dataset_processors as dataset_processors

# Load BERT tokenizer and model (fine-tuned)
config = AutoConfig.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', output_hidden_states =True)
model = AutoModelForSequenceClassification.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', config=config)
tokenizer = AutoTokenizer.from_pretrained('fine-tuned-bert-personality-sentence-segmentation')

labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

def load_model(model_name):
    return {
        'cnn': CNN(5)
    }[model_name]

def label_accuracy(y_true, y_pred):

    # Element-wise comparison to find exact matches
    matches = torch.eq(y_true, y_pred)

    # Calculate accuracy for each label
    label_accuracies = matches.float().mean(dim=0).tolist()

    return label_accuracies

def prepare_data(row):

    # Tokenize sentence
    essays = row['text']

    # Encode them using the tokenizer
    encoded_essay = tokenizer(essays, truncation = True, return_tensors='pt')

    # Convert to embeddings via CLS token
    cls_embedding = []

    with torch.no_grad():
        bert_output = model(**encoded_essay)
        cls_embedding = bert_output.hidden_states[-1][:, 0, :][0]

    # Merge the labels
    merged_labels = [row[key] for key in row.keys() if key in labels]

    # Return the sentence and label
    return cls_embedding, merged_labels

# Load the dataset and pre-process
# Split using K-Fold cross validation (4)
datafile = "data/essays/essays.csv"
dataset = dataset_processors.load_essays_df(datafile)
skf = StratifiedKFold(n_splits=4, shuffle=False)

cls_features = []
merged_labels = []

for index, row in dataset.iterrows():
    bert_output, labels = prepare_data(row)

    cls_features.append(bert_output)
    merged_labels.append(labels)

# Prepare the training parameters
learning_rate = 0.001
batch_size = 32
epochs = 20

print(len(cls_features))
print(merged_labels)

# Load the model required: LSTM, GRU, CNN

# Train the model

# Display the metrics

# Save the model