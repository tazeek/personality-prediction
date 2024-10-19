from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix
from torch.nn import Sigmoid

import torch
import numpy as np

class Evaluator():

    def __init__(self, num_labels, threshold=0.5):

        self._num_labels = num_labels
        self._threshold = threshold

        self._sigmoid = Sigmoid()

    def _save_file_pickle(self, data, file_name):
        ...

    def _save_file_csv(self, data, file_name):
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

        print(gold)
        print(predicted)
        quit()

        for i in range(self._num_labels):
            
            print(i)
            f1 = f1_score(gold[:, i], predicted[:, i])
            accuracy = accuracy_score(gold[:, i], predicted[:, i])
            print(f1)
            print(accuracy)
            quit()

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)

        # Get confusion matrix and save
        confusion_matrices = multilabel_confusion_matrix(gold, predicted)

        return None

    def create_sample_predictions(self, input, predicted):
        ...
