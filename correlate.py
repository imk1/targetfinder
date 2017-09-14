import sys
import scipy.stats as ss
import classify as cl

if __name__ == '__main__':
    X_df, y = cl.load_data(sys.argv[1])
    X = X_df.values
    
    for f1, feature_1 in enumerate(X_df.columns):
        for f2, feature_2 in enumerate(X_df.columns):
            if f1 >= f2:
                continue
            pearson_r, pearson_p = ss.pearsonr(X[:, f1], X[:, f2])
            spearman_r, spearman_p = ss.spearmanr(X[:, f1], X[:, f2])

            fields = [ feature_1, feature_2, pearson_r, pearson_p,
                       spearman_r, spearman_p ]
            print('\t'.join([ str(f) for f in fields ]))
