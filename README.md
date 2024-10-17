
## Improving Personality Detection without the use of external resources

## License
The source code for this project is licensed under the [MIT license](LICENSE.md).

## Major Phases involved

1. Fine-tuning LLMs
2. Dataset Transformation
3. Training model with LLMs
4. Evaluation metrics
5. Visualization of Results

# Execution

For fine-tuning: python fine_tuning_process.py -ss True -pm bert

## Sliding Window Concept

### Fine-tuning

Phases:

1. Load the model to fine-tune
2. Load the dataset to fine-tune
3. Transform and split the dataset
4. Begin fine-tuning the model
5. Save the model

### Training phase

Two phases: Transformation and training

Transformation:

1. Load the dataset
2. (Optional) Perform pre-processing
3. Transform the data using the fine-tuned model
4. Save the data

Training:

1. Load the transformed data
2. Create the model architecture
3. Start training the model
4. Evaluation metrics
5. Save the model

### Saliency of models

Phases:

1. Load the model
2. Create forward hook function
3. Create backward hook function
4. Attach both to the model
5. Fetch the input
6. Perform saliency
7. Decide on the saliency method to perform

---

### Personality and Emotions

Phases:

1. Initialize the model
2. Load the model weights
3. Load the dataset
4. Check the personality for every emotion
5. Convert to label and add it to the dictionary
6. Store it back into the JSON file

### Correlation Analysis

Phases:

1. Load the dataset
2. Go by utterance
3. Check by emotion and the corresponding personalities
4. Create the confusion matrix
5. Normalize the confusion matrix.

### Causation Analysis

Phases:

1. Create the rules for "causation"
2. Go by current and next utterance (unless end)
3. Check by personality
4. Create the diagram (Figure it out)
5. Normalize the diagram (depends on 4)
