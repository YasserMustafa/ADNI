'''From patient_info.py'''

def count_visits(data, modality='', plot=True):
    """
    Keyword Arguments:
    data -- The subset of the data we want visit stats for
    """
    all_visits = np.sort(np.r_[data.VISCODE2.unique(), data.VISCODE.unique()])
    all_visits = np.array([visit for visit in all_visits
                           if str(visit) != 'nan' and\
                           str(visit) != 'f' and\
                           str(visit)[0] != 'v' and\
                           str(visit) != 'nv'])
    all_visits = np.unique(all_visits)
    print "Unique visit codes:", all_visits
    columns = np.r_[['Phase'], ['Count'], ['DXBASELINE'], all_visits]
    rid = data['RID'].unique()
    stats = pd.DataFrame(columns=columns)
    dx_base = get_baseline_classes(data)

    for patient in rid:
        reg_info = DXARM_REG[DXARM_REG['RID'] == patient]
        mode_info = data[data['RID'] == patient]
        values = []
        if mode_info.VISCODE2.isnull().any():
            visits = mode_info['VISCODE']
        else:
            visits = mode_info['VISCODE2']
        if pd.Series('bl').isin(reg_info.VISCODE2).any()\
           or pd.Series('sc').isin(reg_info.VISCODE2).any():
            phase = reg_info[reg_info.VISCODE2 == 'bl'].Phase.unique()[0]
            values.append(phase)
            values.append(len(visits.unique()))
            values.append(dx_base[patient])
            for visit in all_visits:
                if visit in visits.values:
                    values.append('True')
                else:
                    values.append('False')
            stats.loc[patient] = values

    print "Total patients = ", len(stats)
    if plot:
        phases = stats['Phase'].unique()
        counts = []
        for phase in phases:
            counts.append(stats[stats['Phase'] == phase].Count.values)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(counts, bins=np.arange(1, max(stats.Count+2)),
                label=phases.tolist(),
                color=['r', 'g', 'b'][:len(phases)])
        ax.set_xlabel('Number of visits', fontsize=28)
        ax.set_ylabel('Number of patients', fontsize=28)
        ax.set_title('Histogram of patient visits for '+modality+' data',
                     fontsize=30)
        ax.legend()
        ax.yaxis.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.xaxis.set_ticks(np.arange(1, max(stats.Count+2)))
        fig.show()
    return stats

def plot_dx(stats, modality=''):
    """
    Show the distribution of diagnoses against patient-visit counts
    Keyword Arguments:
    stats -- The stats to plot
    """
    dx_base = np.sort(stats['DXBASELINE'].unique())
    counts = []
    for dx in dx_base:
        counts.append(stats[stats['DXBASELINE'] == dx].Count.values)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=np.arange(1, max(stats.Count+2)),
            label=dx_base.tolist())
    ax.set_xlabel('Number of visits', fontsize=28)
    ax.set_ylabel('Number of patients', fontsize=28)
    ax.set_title('Histogram of patient visits for '+modality+' data',
                 fontsize=30)
    ax.legend()
    ax.yaxis.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.xaxis.set_ticks(np.arange(1, max(stats.Count+2)))
    fig.show()
