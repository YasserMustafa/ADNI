"""Read and clean the UCSF Free-surfer data"""

import pandas as pd
import StringIO
import patient_info as pi
import numpy as np

BASE_DIR = '/phobos/alzheimers/adni/'

FDG_FILE = BASE_DIR + 'UCBERKELEYFDG_03_13_14.csv'
AV_FILE = BASE_DIR + 'UCBERKELEYAV45_07_30_14.csv'

FDG = pd.read_csv(StringIO.StringIO(open(FDG_FILE)
                                    .read().replace('\x00', '')))
AV = pd.read_csv(StringIO.StringIO(open(AV_FILE)
                                   .read().replace('\x00', '')))

NORMAL = 1
SMC = 2
EMCI = 3
LMCI = 4
AD = 5

def find_matches():
    """
    find how many patients have data for the same visit in both FDG
    and AV modalities
    """
    merge_on = ['RID', 'VISCODE', 'VISCODE2'] # the columns to merge on
    common = (AV.merge(FDG, on=merge_on, suffixes=['_av', '_fdg'])).merge(
        pi.DXARM_REG, on=merge_on, suffixes=['_pet', '_dx'])

    return common

def get_stats(data, merge=True):
    """
    Keyword Arguments:
    data -- dataframe to generate stats for
    """
    cols = ['NUM_VISITS', 'DXBASELINE',
            'VISIT_1', 'DXCHANGE_1', 'Phase_1',
            'VISIT_2', 'DXCHANGE_2', 'Phase_2']

    if merge:
        data = data.merge(pi.DXARM_REG, on=['RID', 'VISCODE', 'VISCODE2'],
                          suffixes=['_dat', '_dx'])
    rid = data.RID.unique()

    stats = []
    max_size = 0

    for patient in rid:
        info = data[data.RID == patient]
        visits = np.sort(info.VISCODE2.unique())

        row = [len(visits)]
        row.append(info.DXBASELINE.unique()[0])
        for visit in visits:
            patient_dx_vis = info[info.VISCODE2 == visit]
            viscode = patient_dx_vis.VISCODE2.unique()
            dxc = patient_dx_vis.DXCHANGE.unique()
            phase = patient_dx_vis.Phase.unique()

            assert (len(viscode) == 1), 'Error in VISCODE2, RID=%d'%patient
            assert (len(dxc) == 1), 'Error in DXCHANGE, RID=%d'%patient
            #assert (len(phase) == 1), 'Error in Phase, RID=%d'%patient

            row.extend([viscode[0], dxc[0], phase[0]])
        if len(row) > max_size:
            max_size = len(row)

        stats.append(row)
    max_size -= 2
    cols = ['NUM_VISITS', 'DXBASELINE']
    for i in xrange(max_size/3):
        cols.extend(['VISIT_'+str(i+1),
                     'DXCHANGE_'+str(i+1), 'Phase_'+str(i+1)])

    return pd.DataFrame(stats, index=rid, columns=cols)
