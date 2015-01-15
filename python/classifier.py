import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('../viterbi_prob_train.csv')
x = data[data.columns[:-3]]
y = data['DX']

logreg = linear_model.LogisticRegression(penalty='l2',
                                         fit_intercept=True,
                                         C=1e-3)


