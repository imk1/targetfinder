import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from partial_dependence import plot_partial_dependence
from partial_dependence import partial_dependence

import classify as cl

import seaborn as sns

sns.set(font_scale=1.8, style='white',
        rc={ 'text.color':'0', 'xtick.color':'0', 'ytick.color':'0' })

if __name__ == '__main__':
    X, y = cl.load_data(sys.argv[1])

    names = []
    for feature in X.columns:
        feature = feature.replace('enhancer', 'bQTL')
        feature = feature.replace('promoter', 'TSS')
        feature = feature.replace('_', ' ')
        feature = feature.replace('HiC', 'Hi-C')
        names.append(feature)

    clf = cl.get_clf()
    clf.fit(X, y)

    interested = [
#        'BCL3 (promoter)',
#        'H3K4me1 (promoter)',
        'enhancer_distance_to_promoter',
        'HiC_Interact',
#        'CAGE (window)',
#        'H3K4me3 (window)',
#        'DNase-seq (window)',
#        'H2AZ (promoter)',
#        'H3K27me3 (enhancer)',
#        'H3K36me3 (window)',
#        'H3K4me2 (window)',
#        'H3K9ac (window)',
#        'Methylation (window)',
#        'Hi-C_Sum (promoter)',
    ]
    features = []
    for i in interested:
        for idx, feature in enumerate(X.columns):
            if feature == i:
                features.append(idx)
                break
    features.append((features[0], features[1]))
#    features.append((features[3], features[4]))
#    features.append((features[6], features[7]))
    names[features[0]] += ' (bp)'
    names[features[1]] += ' (reads)'
    fig, ax = plot_partial_dependence(clf, X, features,
                                      feature_names=names,
                                      n_jobs=3, grid_resolution=40,
                                      figsize=(20, 6),
                                      contour_kw={'cmap':plt.cm.coolwarm_r})
    fig.suptitle('Partial dependence')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    plt.show()
    fig.savefig('figures/partial_dependence.png')
