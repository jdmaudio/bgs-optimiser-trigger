import numpy as np
import math

class ConfusionMatrix:
    def __init__(self, truth, detections, ROI=None):
        assert truth.shape == detections.shape
        self.truth = truth
        self.detections = detections

        self.ROI = np.ones(truth.shape, dtype=bool) if ROI is None else ROI
        self.compute()

    def compute(self):
        positive_count = np.count_nonzero(np.bitwise_and(self.detections, self.ROI))
        negative_count = np.count_nonzero(np.ones(self.truth.shape[0]) - np.bitwise_and(self.detections, self.ROI))

        temp = np.bitwise_and(np.bitwise_and(self.truth, self.ROI), self.detections)
        self.TP = np.count_nonzero(temp)
        self.FP = positive_count - self.TP

        temp = np.bitwise_not(np.bitwise_and(self.detections, self.ROI))
        temp = np.bitwise_and(self.truth, temp)
        self.FN = np.count_nonzero(temp)
        self.TN = negative_count - self.FN

    def get(self):
        return self.TP, self.TN, self.FP, self.FN

class Metrics:
    def __init__(self, TP, TN, FP, FN):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

        self.accuracy = 0.0
        self.errorRate = 0.0
        self.specificity = 0.0
        self.falsePositiveRate = 0.0
        self.precision = 0.0
        self.MCC = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

        self.computeMetrics()

    def computeMCC(self):
        denominator = (float((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))) + 1e-9
        if denominator <= 0:
            return 0
        else:
            return (float(self.TP) * self.TN - self.FP * self.FN) / math.sqrt(denominator)

    def computeMetrics(self):
        self.accuracy = ((self.TP + self.TN) / float(self.TP + self.TN + self.FP + self.FN))
        self.errorRate = (self.FP + self.FN) / float(self.TP + self.TN + self.FP + self.FN)
        self.specificity = self.FN / float(self.TN + self.FP + 1e-9) 
        self.falsePositiveRate = self.FP / float(self.TN + self.FP)
        self.precision = self.TP / float(self.TP + self.FP + 1e-9)
        self.MCC = self.computeMCC()
        self.recall = self.TP / float(self.TP + self.FN + 1e-9)
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-9)