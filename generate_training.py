#!/usr/bin/env python

import chromatics
import common
import os
import pandas as pd
import sys
import numpy as np

from glob import glob

config_fn = sys.argv[1]
cell_line = config_fn.split('/')[0]
config = common.parse_config(config_fn)
os.chdir(os.path.expanduser(config['working_dir']))

peaks_fn = 'peaks.bed.gz'
methylation_fn = 'methylation.bed.gz'
cage_fn = 'cage.bed.gz'
hic_fn = 'hi-c.bed.gz'
generators = []

# preprocess hi-c
if os.path.exists('../../data/hi-c'):
    hic_df = chromatics.read_bed(glob('../../data/hi-c/sum_all.bed.gz')[0],
                                 names=chromatics.hic_bed_columns,
                                 usecols=chromatics.hic_bed_columns)
    hic_df['name'] = 'Hi-C_Sum'
    chromatics.write_bed(hic_df, hic_fn, compression='gzip')
    generators.append((chromatics.generate_average_signal_features, hic_fn))

## preprocess peaks
if os.path.exists('../../data/peaks'):
    assays = []
    for name, filename, source, accession in pd.read_csv('../../data/peaks/filenames.csv').itertuples(index = False):
        columns = chromatics.narrowpeak_bed_columns if filename.endswith('narrowPeak') else chromatics.broadpeak_bed_columns
        assay_df = chromatics.read_bed('../../data/peaks/{}.gz'.format(filename), names = columns, usecols = chromatics.generic_bed_columns + ['signal_value'])
        assay_df['name'] = name
        assays.append(assay_df)
    peaks_df = pd.concat(assays, ignore_index = True)
    chromatics.write_bed(peaks_df, peaks_fn, compression = 'gzip')
    generators.append((chromatics.generate_average_signal_features, peaks_fn))

# preprocess methylation
if os.path.exists('../../data/methylation'):
    assays = [chromatics.read_bed(_, names = chromatics.methylation_bed_columns, usecols = chromatics.generic_bed_columns + ['mapped_reads', 'percent_methylated']) for _ in glob('../../data/methylation/*.bed.gz')]
    methylation_df = pd.concat(assays, ignore_index = True).query('mapped_reads >= 10 and percent_methylated > 0')
    methylation_df['name'] = 'Methylation'
    del methylation_df['mapped_reads']
    chromatics.write_bed(methylation_df, methylation_fn, compression = 'gzip')
    generators.append((chromatics.generate_average_signal_features, methylation_fn))

# preprocess cage
if os.path.exists('../../data/cage'):
    cage_df = chromatics.read_bed(glob('../../data/cage/*.bed.gz')[0], names = chromatics.cage_bed_columns, usecols = chromatics.cage_bed_columns[:5])
    cage_df['name'] = 'CAGE'
    chromatics.write_bed(cage_df, cage_fn, compression = 'gzip')
    generators.append((chromatics.generate_average_signal_features, cage_fn))

# generate features
pairs_df = pd.read_csv('pairs.csv')
assert pairs_df.duplicated().sum() == 0
training_df = chromatics.generate_training(pairs_df, config['regions'], generators, chunk_size = 2**14, n_jobs = 3)
training_df = training_df.set_index(['enhancer_name', 'promoter_name'])

# add Hi-C interaction status
training_df['HiC_Interact'] = np.zeros(len(training_df['enhancer_end']))
with open('../../data/hi-c/interact_reads.txt', 'r') as ir_file:
    for line in ir_file:
        fields = line.rstrip().split()
        bqtl_name = fields[0] + ':' + fields[1]
        tss_name = fields[2] + ':' + fields[3]
        reads = float(fields[4])
        if (bqtl_name, tss_name) in training_df.index:
            training_df.set_value((bqtl_name, tss_name), 'HiC_Interact', reads)

# save
training_df.to_hdf('training.h5', 'training', mode = 'w', complevel = 1, complib = 'zlib')
