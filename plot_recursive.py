import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(font_scale=1.8, style='white',
        rc={ 'text.color':'0', 'xtick.color':'0', 'ytick.color':'0' })

f1s = []
for pos, line in enumerate(open('bqtls/interpret/recursive_rank.txt')):
    f1s.append(float(line.rstrip().split()[-2]))
f1s = f1s[:15]

fig = plt.figure(figsize=(4, 8))
ax = sns.pointplot(x=np.arange(len(f1s)) + 1, y=f1s, markers='o')

ax.set_xlabel('Number of features')
ax.set_ylabel('F1')
for idx, label in enumerate(ax.xaxis.get_ticklabels()):
    if idx % 4 != 0:
        label.set_visible(False)

plt.tight_layout()
fig.savefig('figures/recursive.png', dpi=500)
plt.show()
