import numpy as np


class KFoldSplit:
    def __init__(self, k, merge_data, random_state=42):
        self.k = k
        self.size = len(merge_data)
        self.valid_size = self.size // k
        self.iteration = 0
        np.random.seed(random_state)
        self.index = np.random.permutation(self.size)      
        self.merge_data = merge_data  
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.k:
            raise StopIteration

        i = self.iteration
        v = self.valid_size
        if i == 0:
            valid_index = self.index[:v]
            train_index = self.index[v:]
        elif i == self.k-1:
            train_index = self.index[:i*v]
            valid_index = self.index[i*v:]
        else:
            valid_index = self.index[i*v:(i+1)*v]
            train_index = np.append(self.index[:i*v], self.index[(i+1)*v:])

        train_datasets = [self.merge_data[i] for i in train_index]
        valid_datasets = [self.merge_data[i] for i in valid_index]

        self.iteration += 1
        return train_datasets, valid_datasets

    def next(self):
        return self.__next__()