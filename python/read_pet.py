"""Read and clean the UCSF Free-surfer data"""

import pandas as pd
import StringIO
from patient_info import clean_visits, get_baseline_classes
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = '/phobos/alzheimers/adni/'

FDG_FILE = BASE_DIR + 'UCBERKELEYFDG_03_13_14.csv'
AV_FILE = BASE_DIR + 'UCBERKELEYAV45_07_30_14.csv'

FDG = pd.read_csv(StringIO.StringIO(open(FDG_FILE)
                                    .read().replace('\x00', '')))
AV = pd.read_csv(StringIO.StringIO(open(AV_FILE)
                                   .read().replace('\x00', '')))
FDG['ROI'] = FDG['ROINAME'] + '_' + FDG['ROILAT']

if 'VISCODE2' in FDG.columns:
    FDG = clean_visits(FDG)
else:
    FDG['VISCODE2'] = FDG['VISCODE']

if 'VISCODE2' in AV.columns:
    AV = clean_visits(AV)
else:
    AV['VISCODE2'] = AV['VISCODE']

def reduce_to_rows():
    """
    Reshape FDG data so that each row represents a visit rather than a
    region

    """
    data = []
    rid = FDG['RID'].unique()
    features = ['MEAN', 'MEDIAN', 'MODE', 'MIN', 'MAX', 'STDEV']
    regions = np.sort(FDG['ROI'].unique())

    for patient in rid:
        row = [patient]
        pet = FDG[FDG['RID'] == patient]
        visits = pet.VISCODE2.unique()
        for visit in visits:
            row.append(visit)
            vis_data = pet[pet['VISCODE2'] == visit].sort('ROI')
            for region in regions:
                row.extend(vis_data[vis_data['ROI'] == region]
                           [features].values.tolist()[0])
            data.append(row)
            row = [patient]

    columns = ['RID', 'VISCODE2']
    for roi in regions:
        columns.extend([roi+'_'+feature for feature in features])

    return pd.DataFrame(data, columns=columns)

def plot_features():
    """
    Show the distribution of the features given the diagnoisis
    """
    features = ['MEAN', 'MEDIAN', 'MODE', 'MIN', 'MAX', 'STDEV']
    regions = FDG['ROI'].unique()
    rid = FDG['RID'].unique()
    dx_base = get_baseline_classes(FDG)
    groups = list(set(dx_base.values()))

    stats = {}
    for feature in features:
        stats[feature] = {}
        for group in groups:
            stats[feature][group] = []

    for patient in rid:
        if patient in dx_base:
            patient_info = FDG[FDG['RID'] == patient]
            visit = np.sort(patient_info['VISCODE2'].unique())[0]
            info = patient_info[patient_info['VISCODE2'] == visit]
            for feature in features:
                stats[feature][dx_base[patient]].append(info[feature].mean())

    for feature in features:
        if 'MCI-REV' in stats[feature]:
            stats[feature].pop('MCI-REV')

        idx = 0
        fig = plt.figure()
        for group in stats[feature].keys():
            idx += 1
            ax = fig.add_subplot(2, 2, idx)
            ax.hist(stats[feature][group], bins=50)
            ax.set_xlabel('Average value')
            ax.set_ylabel('Number of patients')
            ax.set_title('Dx = '+group)
            ax.yaxis.grid(True)

        fig.suptitle('Average of the '+feature+' of the PET value')
        fig.show()

    return stats
