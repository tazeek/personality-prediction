import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
import re
import pickle
import time
import math
import pandas as pd
from pathlib import Path

# add parent directory to the path as well, if running from the finetune folder
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

sys.path.insert(0, os.getcwd())

import utils.gen_utils as utils

def get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer, n_hl, hidden_dim):
    """Read data from pkl file and prepare for training."""

    # Load the file
    file = open(
        inp_dir + dataset + "-" + embed + "-" + embed_mode + "-" + mode + ".pkl", "rb"
    )
    data = pickle.load(file)
    author_ids, data_x, data_y = list(zip(*data))
    file.close()


    # alphaW is responsible for which BERT layer embedding we will be using
    if layer == "all":
        alphaW = np.full([n_hl], 1 / n_hl)

    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1

    # just changing the way data is stored (tuples of minibatches) and
    # getting the output for the required layer of BERT using alphaW
    inputs = []
    targets = []
    n_batches = len(data_y)
    
    for index in range(n_batches):

        # Extracts the CLS token -_-
        # You could have done this in the extraction step :facepalm:
        inputs.extend(np.einsum("k,kij->ij", alphaW, data_x[index]))

        targets.extend(data_y[index])

    inputs = np.array(inputs)
    full_targets = np.array(targets)

    return inputs, full_targets


def training(dataset, inputs, full_targets, inp_dir, save_model, n_classes):
    """Train MLP model for each trait on 10-fold corss-validtion."""

    trait_labels = []

    if dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    n_splits = 10
    fold_acc = {}

    expdata = {}
    expdata["acc"], expdata["trait"], expdata["fold"] = [], [], []

    best_models, best_model, best_accuracy = {}, None, 0.0

    start = time.time()

    for trait_idx in range(full_targets.shape[1]):

        # convert targets to one-hot encoding
        targets = full_targets[:, trait_idx]

        expdata["trait"].extend([trait_labels[trait_idx]] * n_splits)
        expdata["fold"].extend(np.arange(1, n_splits + 1))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        accuracy = []

        for train_index, test_index in skf.split(inputs, targets):

            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = targets[train_index], targets[test_index]

            # converting to one-hot embedding
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

            # Define the neural network architecture            
            model = tf.keras.models.Sequential()

            model.add(
                tf.keras.layers.Dense(50, input_dim=hidden_dim, activation="relu")
            )

            model.add(tf.keras.layers.Dense(n_classes))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=["mse", "accuracy"],
            )

            # Start training
            history = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, y_test),
                verbose=0,
            )

            max_val_accuracy = max(history.history["val_accuracy"])
            accuracy.append(max_val_accuracy)
            expdata["acc"].append(100 * max_val_accuracy)

            # check if the current model is the best so far
            if max_val_accuracy > best_accuracy:
                best_accuracy = max_val_accuracy
                best_model = model

        # store the best model for this trait
        average = sum(accuracy) / len(accuracy)
        print(f'Average for {trait_labels[trait_idx]}: {average}')
        print("\n\n")
        best_models[trait_labels[trait_idx]] = best_model

    finish = time.time() - start
    print(f'Time taken to train: {finish} seconds')
    print("\n\n")
    
    # save the best models to separate files
    if str(save_model).lower() == "yes":
        path = inp_dir + "finetune_mlp_lm"
        Path(path).mkdir(parents=True, exist_ok=True)

        for trait_label, best_model in best_models.items():
            best_model.save(f"{path}/MLP_LM_{trait_label}.h5")

    print(expdata)
    df = pd.DataFrame.from_dict(expdata)
    return df


def logging(df, log_expdata=True):
    """Save results and each models config and hyper parameters."""
    (
        df["network"],
        df["dataset"],
        df["lr"],
        df["batch_size"],
        df["epochs"],
        df["model_input"],
        df["embed"],
        df["layer"],
        df["mode"],
        df["embed_mode"],
        df["jobid"],
    ) = (
        network,
        dataset,
        lr,
        batch_size,
        epochs,
        MODEL_INPUT,
        embed,
        layer,
        mode,
        embed_mode,
        jobid,
    )

    pd.set_option("display.max_columns", None)
    print(df.head(5))

    # save the results of our experiment
    if log_expdata:
        Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path + "expdata.csv"):
            df.to_csv(path + "expdata.csv", mode="a", header=True)
        else:
            df.to_csv(path + "expdata.csv", mode="a", header=False)


if __name__ == "__main__":

    # Get the args
    (
        inp_dir,
        dataset,
        lr,
        batch_size,
        epochs,
        log_expdata,
        embed,
        layer,
        mode,
        embed_mode,
        jobid,
        save_model,
    ) = utils.parse_args()
    # embed_mode {mean, cls}
    # mode {512_head, 512_tail, 256_head_tail}

    network = "MLP"
    MODEL_INPUT = "LM_features"
    print("{} : {} : {} : {} : {}".format(dataset, embed, layer, mode, embed_mode))
    
    n_classes = 2
    seed = jobid
    np.random.seed(seed)
    tf.random.set_seed(seed)

    start = time.time()
    path = "explogs/"

    if re.search(r"base", embed):
        n_hl = 12
        hidden_dim = 768

    elif re.search(r"large", embed):
        n_hl = 24
        hidden_dim = 1024

    # Load input data
    inputs, full_targets = get_inputs(
        inp_dir, dataset, embed, 
        embed_mode, mode, layer,
        n_hl, hidden_dim
    )

    # Perform the training
    df = training(
        dataset, inputs, full_targets, 
        inp_dir, save_model, n_classes)
    
    # Save
    logging(df, log_expdata)
