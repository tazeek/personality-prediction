

class Evaluator():

    def __init__(self):
        ...

    def _save_file_pickle(self, data, file_name):
        ...

    def _save_file_csv(self, data, file_name):
        ...

    def confusion_matrix(self, gold, predicted):
        ...

    def calculate_accuracy(self, gold, predicted):
        ...

    def calculate_f1(self, gold, predicted):
        ...

    def create_sample_predictions(self, input, predicted):
        ...
