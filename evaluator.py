from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix
from torch.nn import Sigmoid

import torch
import pickle
import numpy as np

class Evaluator():

    def __init__(self, file_name, labels, threshold=0.5):

        self._labels = labels
        self._threshold = threshold
        self._file_name = file_name

        self._sigmoid = Sigmoid()

    def _save_file_pickle(self, data):

        with open(f'{self._file_name}_cm_data.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_file_csv(self, data):
        ...

    def convert_predictions(self, pred_logits):
        probs = self._sigmoid(pred_logits)

        # Convert predictions to integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= self._threshold)] = 1

        return y_pred

    def calculate_scores(self, gold, predicted):

        f1_scores = []
        accuracy_scores = []

        gold = np.array(gold)
        predicted = np.array(predicted)

        for i, label in enumerate(self._labels):

            f1 = f1_score(gold[:, i], predicted[:, i])
            accuracy = accuracy_score(gold[:, i], predicted[:, i])

            print(f"{label}: (Accuracy - {accuracy:.4f}), (F1 - {f1:.4f})")
            print("\n")

        # Get confusion matrix and save
        confusion_matrices = multilabel_confusion_matrix(gold, predicted)
        self._save_file_pickle(confusion_matrices)

        return None

    def create_sample_predictions(self, input, predicted):
        ...
