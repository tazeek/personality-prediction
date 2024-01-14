from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# TODO:
# - Load the pre-trained model
# - Load the DailyDialog dataset
# - Go by utterance
# - Save in pickle format (file name: {file_type}_{model}_{finetuned})

def load_finetuned_bert(model_name):
    config = AutoConfig.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', output_hidden_states =True)
    model = AutoModelForSequenceClassification.from_pretrained('fine-tuned-bert-personality-sentence-segmentation', config=config)
    tokenizer = AutoTokenizer.from_pretrained('fine-tuned-bert-personality-sentence-segmentation')

    return model, tokenizer

# Load BERT tokenizer and model (fine-tuned)
model_name = 'fine-tuned-bert-personality-sentence-segmentation'
model, tokenizer = load_finetuned_bert(model_name)