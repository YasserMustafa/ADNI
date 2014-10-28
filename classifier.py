"""Build a classifier to make the predictions"""

import pandas as pd
import read_pet as pet
import read_mri as mri
import patient_info as pi
import numpy as np
from sklearn import cross_validation, svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pdb
from random import shuffle

LABELS = {'nl':1, 'mci':2, 'ad':3}

def generate_features_fdg_bl():
    """
    Generate a feature vector for each sample of the fdg data
    """
    data = pet.FDG
    data['REGION'] = data['ROINAME'] + data['ROILAT']
    dx_base = pi.get_baseline_classes(data, 'ADNI1')
    features = ['MEAN', 'MEDIAN', 'MODE', 'MIN', 'MAX', 'STDEV']
    X = []
    y = []

    for patient in dx_base.keys():
        readings = data[(data['RID'] == patient) &
                        (data['VISCODE'] == 'bl')]
        if not readings.empty:
            assert (len(readings) == 5),\
                'More than one baseline reading for RID=%d'%patient
            patient_features = readings.pivot(index='RID',
                                              columns='REGION')\
                [features].values[0]
            X.append(patient_features)
            y.append(LABELS[dx_base[patient]])

    return np.array(X), np.array(y)


def generate_features_mri_bl():
    """
    Generate a feature vector for each patient with a baseline MRI scan
    """
    data = mri.FSX
    data = data[data['STATUS'] == 'complete']
    dx_base = pi.get_baseline_classes(data, 'ADNI1')
    features = [col for col in data.columns
                if col[:2] == 'ST' and\
                not col == 'STATUS' and\
                ~pd.isnull(data[col]).any()]

    X = []
    Y = []
    for patient in dx_base.keys():
        readings = data[(data['RID'] == patient) &
                        (data['VISCODE'] == 'sc')][:1]
        if not readings.empty:
            assert (len(readings) == 1),\
                'More than one baseline reading for RID=%d'%patient
            patient_features = readings[features].values[0]
            X.append(patient_features)
            Y.append(LABELS[dx_base[patient]])

    return np.array(X), np.array(Y)

def classify(X, Y):
    """
    Classify patients based on FDG-PET features
    """
    two_class = True
    classes = ['nl', 'ad']
    if two_class:
        first = [i for i in xrange(len(Y)) if Y[i] == LABELS[classes[0]]]
        second = [i for i in xrange(len(Y)) if Y[i] == LABELS[classes[1]]]
        wanted_idx = first + second
        shuffle(wanted_idx)
        X = X[wanted_idx]
        Y = Y[wanted_idx]

    n_folds = 10
    kfold = cross_validation.KFold(len(X), n_folds=n_folds, shuffle=True)
    clf = svm.LinearSVC(verbose=1, C=.02, tol=1e-5,
                        fit_intercept=True, loss='l1')
    #clf = svm.SVC(verbose=1, C=0.5, kernel='rbf')

    total_acc = 0
    idx = 0
    # classify NL vs AD
    for train_idx, test_idx in kfold:
        idx += 1
        print "\nFold", idx

        scaler = StandardScaler(with_mean=True, with_std=True).fit(X[train_idx])
        X = scaler.transform(X)

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train, Y_train)
        pred = clf.predict(X_test)

        train_acc = sum(Y_train == clf.predict(X_train))*1.0/len(X_train)
        test_acc = sum(Y_test == pred)*1.0/len(X_test)
        print "Training accuracy: ", train_acc
        print "Testing accuracy: ", test_acc
        #print "Coefficients: ", clf.coef_
        total_acc += test_acc

    total_acc = total_acc*1.0/n_folds
    print "\nAverage testing accuracy: ", total_acc
    print "\n"

if __name__ == '__main__':
    X, Y = generate_features_mri_bl()
    #X, Y = generate_features_fdg_bl()
    print "Running classifier..."
    classify(X, Y)
