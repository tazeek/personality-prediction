from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix

class Evaluator():

    def __init__(self, num_labels, threshold=0.5):

        self._num_labels = num_labels
        self._threshold = threshold

    def _save_file_pickle(self, data, file_name):
        ...

    def _save_file_csv(self, data, file_name):
        ...

    def confusion_matrix(self, gold, predicted):
        ...

    def calculate_scores(self, gold, predicted):

        f1_scores = []
        accuracy_scores = []

        for i in range(self.num_labels):

            f1_score = f1_score(gold[:, i], predicted[:, i])
            accuracy_score = accuracy_score(gold[:, i], predicted[:, i])

            accuracy_scores.append(accuracy_score)
            f1_scores.append(f1_score)

        return None

    def create_sample_predictions(self, input, predicted):
        ...
