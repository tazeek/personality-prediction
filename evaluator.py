from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix
from torch.nn import Sigmoid

import torch
import numpy as np

class Evaluator():

    def __init__(self, num_labels, threshold=0.5):

        self._num_labels = num_labels
        self._threshold = threshold

        self._sigmoid = Sigmoid()

    def _convert_predictions(self, pred_logits):
        probs = self._sigmoid(torch.Tensor(pred_logits))

        # Convert predictions to integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= self._threshold)] = 1

        return y_pred

    def _save_file_pickle(self, data, file_name):
        ...

    def _save_file_csv(self, data, file_name):
        ...

    def calculate_scores(self, gold, predicted):

        f1_scores = []
        accuracy_scores = []

        predicted = self._convert_predictions(predicted)

        for i in range(self.num_labels):

            f1_score = f1_score(gold[:, i], predicted[:, i])
            accuracy_score = accuracy_score(gold[:, i], predicted[:, i])

            accuracy_scores.append(accuracy_score)
            f1_scores.append(f1_score)

        # Get confusion matrix and save
        confusion_matrices = multilabel_confusion_matrix(gold, predicted)

        return None

    def create_sample_predictions(self, input, predicted):
        ...
