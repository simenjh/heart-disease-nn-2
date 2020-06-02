import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def read_dataset(data_file):
    dataset = pd.read_csv(data_file)
    return dataset.values


def preprocess(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)
    return X, y


def standardize(X1, *args):
    sc = StandardScaler()
    Xs_standard = sc.fit_transform(X1)
    if args != ():
        Xs_standard = [Xs_standard]
        Xs_standard.extend([sc.transform(Xi) for Xi in args])
    return Xs_standard
    
