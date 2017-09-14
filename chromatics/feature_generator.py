#!/usr/bin/env python

import sys
import chromatics
import gzip
import numpy as np
import pandas as pd
import sklearn.externals.joblib as joblib
import pybedtools as bt
from genomedatalayer.genomedatalayer import extractors

def get_bigwig_features(chunk_df, region, dataset):
	# Get values at a position from a bigwig file
	assert (chunk_df[region + '_end'] > chunk_df[region + '_start']).all()
	bwe = extractors.BigwigExtractor(dataset)
	signalsDict = {region + '_name': chunk_df[region + '_name'], region + '_chrom': chunk_df[region + '_chrom'], region + '_position': chunk_df[region + '_position'], 'dataset': [dataset + " (bigwig Signals)"] * len(chunk_df.index), 'signals': np.zeros((len(chunk_df.index),), dtype=np.float)}
	signals_df = pd.DataFrame(signalsDict)
	i = 0
	for index, row in chunk_df.iterrows():
		# Iterate through the positions and create intervals for each
		position = (row[region + '_chrom'], row[region + '_position'])
		interval = bt.Interval(position[0], position[1] - 1, position[1])
		perPositionSignal = bwe([interval])
		signals_df.set_value(i, 'signals', abs(perPositionSignal[i][0][0][0]))
		i = i + 1
	assert(not(signals_df.empty))
	return signals_df.pivot_table(index = region + '_name', columns = 'dataset', values = 'signals')
	
def get_average_bigwig_features(chunk_df, region, (dataset, regionWidth)):
	# Get average values in a region from a bigwig file
	assert (chunk_df[region + '_end'] > chunk_df[region + '_start']).all()
	bwe = extractors.BigwigExtractor(dataset)
	signalsDict = {region + '_name': chunk_df[region + '_name'], region + '_chrom': chunk_df[region + '_chrom'], region + '_start': chunk_df[region + '_start'], region + '_end': chunk_df[region + '_end'], 'dataset': [dataset + " (Average bigwig Signals)"] * len(chunk_df.index), 'averageSignals': np.zeros((len(chunk_df.index),), dtype=np.float)}
	signals_df = pd.DataFrame(signalsDict)
	i = 0
	for index, row in chunk_df.iterrows():
		# Iterate through the positions and create intervals for each
		regionInfo = (row[region + '_chrom'], row[region + '_start'], row[region + '_end'])
		interval = bt.Interval(regionInfo[0], regionInfo[1], regionInfo[2])
		perPositionSignal = bwe([interval])
		signals_df.set_value(i, 'averageSignals', np.mean(np.abs(perPositionSignal[i][0][0])))
		i = i + 1
	assert(not(signals_df.empty))
	return signals_df.pivot_table(index = region + '_name', columns = 'dataset', values = 'averageSignals')

def getLocation(line, positionCol, offset=True):
	# Get the position from the current line and make it into a pybedtool Interval
	lineElements = line.strip().split()
	otherPosition = (lineElements[0], int(lineElements[positionCol]))
	if offset:
		# The position has been recorded as a chromosome and an offset
		otherPosition = (lineElements[0], int(lineElements[1]) + int(lineElements[positionCol]))
	return bt.Interval(otherPosition[0], otherPosition[1] - 1, otherPosition[1])
	
def show_value(s):
	"""
	Convert unicode to str under Python 2;
	all other values pass through unchanged
	"""
	if sys.version_info.major == 2:
		if isinstance(s, unicode):
			return str(s)
	return s

def get_closest_position_features(chunk_df, region, (dataset, maxDistance, logPositions, positionCol, offset)):
	# For each position in the chunk, find the closest position in the dataset
	# ASSUMES THAT dataset IS SORTED BY CHROMOSOME, POSITION
	assert (chunk_df[region + '_end'] > chunk_df[region + '_start']).all()
	otherPositionsFile = gzip.open(dataset)
	otherPositionsList = bt.BedTool([getLocation(line, positionCol, offset=offset) for line in otherPositionsFile]).sort()
	otherPositionsFile.close()
	distancesDict = {region + '_name': chunk_df[region + '_name'], region + '_chrom': chunk_df[region + '_chrom'], region + '_position': chunk_df[region + '_position'], 'dataset': [dataset + " (Position Distances)"] * len(chunk_df.index), 'distances': np.ones((len(chunk_df.index),), dtype=np.float16) * maxDistance}
	distances_df = pd.DataFrame(distancesDict)
	intervalList = []
	
	for index, row in chunk_df.iterrows():
		# Iterate through the positions and make an Interval for each
		intervalList.append(bt.Interval(row[region + '_chrom'], row[region + '_position'] - 1, row[region + '_position']))
	positionsList = bt.BedTool(intervalList)
	closestPositionsList = positionsList.closest(otherPositionsList, d=True, t="first")
	
	index = 0
	for cp in closestPositionsList:
		# Iterate through the closest positons and set the value in the data frame to the closest position
		if logPositions:
			# Use the log2 of the distance
			distances_df.set_value(index, 'distances', np.log2(int(show_value(cp[12])) + 1))
		else:
			# Use the distance
			distances_df.set_value(index, 'distances', int(show_value(cp[12])))
		index = index + 1

	assert(not(distances_df.empty))
	return distances_df.pivot_table(index = region + '_name', columns = 'dataset', values = 'distances')

def generate_intersection_features(chunk_df, region, dataset):
	assert (chunk_df[region + '_end'] > chunk_df[region + '_start']).all()
	regionFile = gzip.open(dataset)
	regionList = bt.BedTool(dataset).sort()
	intersectionDict = {region + '_name': chunk_df[region + '_name'], region + '_chrom': chunk_df[region + '_chrom'], region + '_start': chunk_df[region + '_start'], region + '_end': chunk_df[region + '_end'], 'dataset': [dataset + " (Region Intersections)"] * len(chunk_df.index), 'intersections': np.zeros((len(chunk_df.index),), dtype=np.int)}
	intersections_df = pd.DataFrame(intersectionDict)
	intervalList = []
	
	for index, row in chunk_df.iterrows():
		# Iterate through the rows and make an interval for each
		intervalList.append(bt.Interval(row[region + '_chrom'], row[region + '_position'] - 1, row[region + '_position']))
	positionsList = bt.BedTool(intervalList)
	intersectionsList = positionsList.intersect(regionList, c=True, f=0.5)
	
	index = 0
	for intersection in intersectionsList:
		# Iterate through the rows and find those that intersect the region
		intersections_df.set_value(index, 'intersections', int(show_value(intersection[6])))
		index = index + 1
			
	assert(not(intersections_df.empty))
	return intersections_df.pivot_table(index = region + '_name', columns = 'dataset', values = 'intersections')

def generate_average_signal_features(chunk_df, region, dataset):
	assert (chunk_df[region + '_end'] > chunk_df[region + '_start']).all()

	region_bed_columns = ['{}_{}'.format(region, _) for _ in chromatics.generic_bed_columns]
	signal_df = chromatics.bedtools('intersect -wa -wb', chunk_df[region_bed_columns].drop_duplicates(region + '_name'), dataset, right_names = chromatics.signal_bed_columns)

	group_columns = ['{}_{}'.format(region, _) for _ in ['name', 'start', 'end']] + ['dataset']
	average_signal_df = signal_df.groupby(group_columns, sort = False, as_index = False).aggregate({'signal_value': sum})
	average_signal_df['signal_value'] /= average_signal_df[region + '_end'] - average_signal_df[region + '_start']
	average_signal_df['dataset'] += ' ({})'.format(region)
	assert(not(average_signal_df.empty))

	return average_signal_df.pivot_table(index = region + '_name', columns = 'dataset', values = 'signal_value')

def generate_chunk_features(pairs_df, regions, generators, chunk_size, chunk_number, max_chunks):
    print(chunk_number, max_chunks - 1)

    chunk_lower_bound = chunk_number * chunk_size
    chunk_upper_bound = chunk_lower_bound + chunk_size
    chunk_df = pairs_df.iloc[chunk_lower_bound:chunk_upper_bound]
    assert 0 < len(chunk_df) <= chunk_size

    index_columns = ['{}_name'.format(region) for region in regions]
    features_df = chunk_df[index_columns]
    for region in regions:
        region_features = [generator(chunk_df, region, dataset) for generator, dataset in generators]
        region_features_df = pd.concat(region_features, axis = 1)
        features_df = pd.merge(features_df, region_features_df, left_on = '{}_name'.format(region), right_index = True, how = 'left')
    return features_df.set_index(index_columns)

def generate_training(pairs_df, regions, generators, chunk_size = 2**20, n_jobs = 1):
	for region in regions:
		region_bed_columns = {'{}_{}'.format(region, _) for _ in chromatics.generic_bed_columns}
		assert region_bed_columns.issubset(pairs_df.columns)

	max_chunks = int(np.ceil(float(len(pairs_df)) / float(chunk_size)))
    
	features_df = pd.DataFrame()
	if max_chunks == 1:
		# There is only 1 chunk, so do not use parallization
		results = generate_chunk_features(pairs_df, regions, generators, chunk_size, 0, max_chunks)
		features_df = results.fillna(0)
	else:
		# There are multiple chunks, so generate features for them in parallele
		results = joblib.Parallel(n_jobs)(
			joblib.delayed(generate_chunk_features)(pairs_df, regions, generators, chunk_size, chunk_number, max_chunks)
			for chunk_number in range(max_chunks))
		features_df = pd.concat(results).fillna(0)
    
	training_df = pd.merge(pairs_df, features_df, left_on = ['{}_name'.format(region) for region in regions], right_index = True)
	if not training_df.index.is_unique:
		# If an index is not unique, print that index for debugging purposes
		print training_df.index
	assert training_df.index.is_unique
	assert training_df.columns.is_unique
	return training_df

def get_random_pairs(pair_count, region_a_prefix = 'r1', region_b_prefix = 'r2', random_state = 0):
    random_state = np.random.RandomState(random_state)
    f1_start = random_state.randint(0, 1e6, pair_count)
    f2_start = random_state.randint(0, 1e6, pair_count)
    pair_coordinates = [
        random_state.choice(chromatics.chroms, pair_count),
        f1_start,
        f1_start + random_state.randint(1, 50000, pair_count),
        ['{}_{}'.format(region_a_prefix, _) for _ in range(pair_count)],
        random_state.choice(chromatics.chroms, pair_count),
        f2_start,
        f2_start + random_state.randint(1, 50000, pair_count),
        ['{}_{}'.format(region_b_prefix, _) for _ in range(pair_count)]
        ]
    pair_columns = ['{}_{}'.format(region_a_prefix, _) for _ in chromatics.generic_bed_columns] + \
        ['{}_{}'.format(region_b_prefix, _) for _ in chromatics.generic_bed_columns]
    return pd.DataFrame(dict(zip(pair_columns, pair_coordinates)), columns = pair_columns)

def test_generate_average_signal_features():
    enhancers_df = chromatics.read_bed('enhancers.bed', names = chromatics.enhancer_bed_columns)
    average_signal_df = generate_average_signal_features(enhancers_df, 'enhancer', 'peaks.bed')
    assert average_signal_df.loc['enhancer1', 'RAD21 (enhancer)'] == 2.1
    assert average_signal_df.loc['enhancer1', 'CTCF (enhancer)'] == 0.502
    print(average_signal_df)

def test_generate_training():
    regions = ['enhancer', 'promoter']
    pairs_df = get_random_pairs(100, regions[0], regions[1])
    print(pairs_df.head())

    signal_df = chromatics.read_bed('wgEncodeAwgDnaseUwdukeK562UniPk.narrowPeak.gz', names = chromatics.narrowpeak_bed_columns, usecols = ['chrom', 'start', 'end', 'signal_value'])
    signal_df['dataset'] = 'DNase'
    signal_df = signal_df[chromatics.signal_bed_columns]
    generators = [(generate_average_signal_features, signal_df)]
    training_df = generate_training(pairs_df, regions, generators, chunk_size = len(pairs_df) // 2)
    print(training_df.head(), '\n')

if __name__ == '__main__':
    test_generate_average_signal_features()
    test_generate_training()
