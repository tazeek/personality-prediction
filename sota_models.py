from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sota_list import *

import utils.dataset_processors as dataset_processors

# Load BERT tokenizer and model (fine-tuned)
model = AutoModelForSequenceClassification.from_pretrained('fine-tuned-bert-personality-sentence-segmentation')
tokenizer = AutoTokenizer.from_pretrained('fine-tuned-bert-personality-sentence-segmentation')

# Load the dataset and pre-process
# Split using K-Fold cross validation (4)
datafile = "data/essays/essays.csv"
dataset = dataset_processors.load_essays_df(datafile)
skf = StratifiedKFold(n_splits=4, shuffle=False)

# Prepare the training parameters
learning_rate = 0.001
batch_size = 32
epochs = 20

# Load the model required: LSTM, GRU, CNN

# Train the model

# Display the metrics

# Save the model