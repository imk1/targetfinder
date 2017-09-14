import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from math import log10

sns.set(font_scale=1.6, style='white',
        rc={ 'text.color':'0', 'xtick.color':'0', 'ytick.color':'0' })

def pretty_features(features):
    names = []
    for feature in features:
        feature = feature.replace('enhancer', 'bQTL')
        feature = feature.replace('promoter', 'TSS')
        feature = feature.replace('_', ' ')
        feature = feature.replace('HiC', 'Hi-C')
        names.append(feature)
    return names

p_df = pd.DataFrame()

for line in open(sys.argv[1], 'r'):
    fields = line.rstrip().split('\t')
    feature0 = fields[0]
    feature1 = fields[1]
    p = float(fields[2])

    if len(p_df.index) < 20 or feature0 in p_df.index:
        if len(p_df.columns) < 20 or feature1 in p_df.columns:
            p_df = p_df.set_value(feature0, feature1, -log10(p))
p_df = p_df.fillna(value=0)

# Generate a mask for the upper triangle
#mask = np.zeros_like(p_df, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
            
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and pect aspect ratio
sns.heatmap(p_df, cmap=plt.cm.Greens,
            square=True, linewidths=.5, cbar_kws={"shrink": .8,
                                                  'label':'Mutually informative features, -log10(p)'}, 
            ax=ax,
            xticklabels=pretty_features(p_df.index),
            yticklabels=pretty_features(p_df.columns))
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment('right')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
fig.savefig('figures/heatmap_p.png', dpi=500)
plt.show()
