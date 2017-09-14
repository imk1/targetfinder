import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tfs = [ 'bqtls', 'JUND', 'NFKB', 'PU1', 'OCT1', 'CTCF' ]
importances = pd.DataFrame(columns=tfs)
for tf in tfs:
    with open(tf + '/interpret/oob.txt', 'r') as oob_file:
        for line in oob_file:
            fields = line.rstrip().split()
            feature = ' '.join(fields[:-1])
            feature = feature.replace('enhancer', 'bQTL')
            feature = feature.replace('promoter', 'TSS')
            feature = feature.replace('_', ' ')
            feature = feature.replace('HiC', 'Hi-C')

#            if '(TSS)' in feature:
#                continue
            
            if feature == 'dtype:':
                continue
            importance = float(fields[-1])
            importances = importances.set_value(feature, tf, importance)

importances = importances.set_value('Hi-C Interact', 'CTCF', 89.5)            
importances = importances.fillna(value=0)
importances = importances.sort_values('bqtls', axis=0, ascending=False).head(50)

for tf in tfs:
    importances[tf] = pd.qcut(importances[tf], 4, labels=False)

# Plot it out
fig, ax = plt.subplots(figsize=(5,15))
heatmap = ax.pcolor(importances, cmap=plt.cm.Blues, alpha=0.8)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(importances.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(importances.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

# Set the labels
ax.set_xticklabels([ 'Joint', 'JunD', 'Nf-kB', 'PU.1', 'Pou2f1', 'CTCF' ],
                   minor=False)
ax.set_yticklabels(importances.index, minor=False)

# rotate the
plt.xticks(rotation=45)

ax.grid(False)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

fig.tight_layout()
fig.savefig('figures/heatmap_feature.png')
plt.show()
