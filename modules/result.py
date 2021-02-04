import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

class TestResult:
    def __init__(self, pred, true, soft):
        self.pred = pred
        self.true = true
        self.soft = soft
        
        self.mat = confusion_matrix(true, pred)
        self.supports = self.mat.sum(axis=1)
        self.total = self.mat.sum()
        
        FP = self.mat.sum(axis=0) - np.diag(self.mat) 
        FN = self.mat.sum(axis=1) - np.diag(self.mat)
        TP = np.diag(self.mat)
        TN = self.total - (FP + FN + TP)
        self.FP = FP.astype(float)
        self.FN = FN.astype(float)
        self.TP = TP.astype(float)
        self.TN = TN.astype(float)
        
        self.accuracy = np.diag(self.mat).sum().item() / len(true)
        
        self.specificity = self.TN / (self.TN+self.FP)
        self.sensitivity = self.TP / (self.TP+self.FN)
        
        
    def print_result(self):
        print(f'Accuracy: {self.accuracy*100:.2f} %')
        print('')

        print('Confusion Matrix:\n', self.mat)
        print('')

        print('   class\tsensitivity\tspecificity\tsupport')
        print('')

        print(f'   covid\t{self.sensitivity[0]:.2f}\t\t{self.specificity[0]:.2f}\t\t{self.supports[0]}')
        print(f' healthy\t{self.sensitivity[1]:.2f}\t\t{self.specificity[1]:.2f}\t\t{self.supports[1]}')
        print(f'  others\t{self.sensitivity[2]:.2f}\t\t{self.specificity[2]:.2f}\t\t{self.supports[2]}')
        print('')

        macro_sensitivity = sum(self.sensitivity) / 3.0
        macro_specificity = sum(self.specificity) / 3.0
        print(f'   macro\t{macro_sensitivity:.2f}\t\t{macro_specificity:.2f}\t\t{self.total}')

        weighted = [support / self.total for support in self.supports]
        weighted_sensitivity = weighted @ self.sensitivity
        weighted_specificity = weighted @ self.specificity
        print(f'weighted\t{weighted_sensitivity:.2f}\t\t{weighted_specificity:.2f}\t\t{self.total}')
        print('')
        
    def majority_vate_by_patient(self, test_loader, name=''):
        images_path = test_loader.dataset.samples
        # images_path -> [ [images path, label] * 835 ]

        with open(f"majority_{name}.csv", "w") as f:
            wr = csv.writer(f)
            wr.writerow(["file", "prob_0", "prob_1", "prob_2", "pred", "label"])
            for i in range(len(self.pred)):
                f = os.path.basename(images_path[i][0])
                prob_0 = round(self.soft[i][0], 6)
                prob_1 = round(self.soft[i][1], 6)
                prob_2 = round(self.soft[i][2], 6)
                prediction = self.pred[i]
                label = self.true[i]
                wr.writerow([f, prob_0, prob_1, prob_2, prediction, label]) 