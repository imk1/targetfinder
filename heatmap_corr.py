import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import log10

sns.set(font_scale=1.8, style='white',
        rc={ 'text.color':'0', 'xtick.color':'0', 'ytick.color':'0' })

features = [ line.rstrip() for line in open('./sig_features.txt', 'r') ][0:15]
corrs_df = pd.DataFrame()
names = []
for feature in features:
    feature = feature.replace('enhancer', 'bQTL')
    feature = feature.replace('promoter', 'TSS')
    feature = feature.replace('_', ' ')
    feature = feature.replace('HiC', 'Hi-C')
    names.append(feature)

for feature in features:
    corrs_df = corrs_df.set_value(feature, feature, 0)

for line in open(sys.argv[1], 'r'):
    fields = line.rstrip().split('\t')
    feature0 = fields[0]
    feature1 = fields[1]
    corr = float(fields[4])

    if feature0 in features:
        if feature1 in features:
            corrs_df = corrs_df.set_value(feature1, feature0, corr)
corrs_df = corrs_df.fillna(value=0)

# Generate a mask for the upper triangle
mask = np.zeros_like(corrs_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
            
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
with sns.axes_style("white"):
    sns.heatmap(corrs_df, mask=mask, cmap=cmap,
               square=True, linewidths=.5, cbar_kws={"shrink": .8,
                                                     'label':'Correlation'},
               ax=ax, xticklabels=names, yticklabels=names)
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment('right')
plt.xticks(rotation=38)
plt.yticks(rotation=0)
plt.tight_layout()

fig.savefig('figures/heatmap_corr.png', dpi=500)
plt.show()
