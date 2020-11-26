import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

dirname = os.path.dirname(__file__)
dimensions = ["anger", "fear", "joy", "sadness", "disgust", "positiveSurprise",
              "negativeSurprise", "anticipation", "trust"]

class Rating:
    def __init__(self, name):
        path = dirname + f"/{name}.csv"
        self.name = name
        self.data = pd.read_csv(path, names=dimensions)
        self.normalized_frame = pd.DataFrame(columns=dimensions)
        self.results = self.result_dict()

    # create a dictionary containing the results
    # (emotions and their respective ratings)
    def result_dict(self, frame=None):
        if frame is None:
            frame = self.data
        result_dict = dict()
        for (column_name, column_data) in frame.iteritems():
            result_dict[column_name] = column_data.to_list()
        return result_dict

    # normalize values, either 'binary' (value > 0: 1, else 0) or
    # 'intensity' (value > 1: 1, else 0)
    def normalize_values(self, style="binary"):
        if style == "binary":
            for emotion in dimensions:
                normalized_column = np.where(self.data[emotion] > 0, 1, 0)
                self.normalized_frame[emotion] = normalized_column
        elif style == "intensity":
            for emotion in dimensions:
                normalized_column = np.where(self.data[emotion] > 1, 1, 0)
                self.normalized_frame[emotion] = normalized_column
        else:
            return "FAILED. " \
                   "Please pass 'binary' or 'intensity' to type argument."

    # calculate kappa score for all dimensions (columns of a DataFrame)
    def calculate_kappas(self, other, normalized=False):
        kappas = dict()
        for emotion in dimensions:
            if normalized == False:
                kappa = round(cohen_kappa_score(
                    self.data[emotion], other.data[emotion]), 2)
            elif normalized:
                kappa = round(cohen_kappa_score(
                    self.normalized_frame[emotion],
                    other.normalized_frame[emotion]), 2)
            else:
                return "FAILED. " \
                       "Please pass a boolean value to normalized argument."
            if 0 <= kappa <= 1:
                kappas[emotion] = kappa
            else:
                kappa = 0
        return f"Cohen's kappa score for agreement between {self.name} and " \
               f"{other.name}: {kappas}. Mean value: " \
               f"{(sum(kappas.values())/len(kappas)):.2f}"

    """
    # ( CURRENTLY NOT IN USE )
    # compare columns of two annotators' dataframes with respect to overlaps
    # and return a column containing binary information about matches
    def compare(self, emotion, other, frame=None):
        if frame == None:
            frame = self.data
        comparison_column = \
            np.where(frame.data[emotion] == other.data[emotion], 1, 0)
        return comparison_column
    """


# initialize Rating instances for each annotator's csv file
alex_frame, sarina_frame, xanat_frame = \
    Rating("alex"), Rating("sarina"), Rating("xanat")

# 'normalize' ratings to binary values for processing
alex_frame.normalize_values()
sarina_frame.normalize_values()
xanat_frame.normalize_values()

# kappa results
# alexs/sarina
print(alex_frame.calculate_kappas(sarina_frame, normalized=True))
# alex/xanat
print(alex_frame.calculate_kappas(xanat_frame, normalized=True))
# sarina/xanat
print(sarina_frame.calculate_kappas(xanat_frame, normalized=True))