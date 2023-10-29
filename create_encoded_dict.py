import pandas as pd
from tqdm import tqdm
import json
import sys
import random
import pickle
import argparse

DEF_DATA_PATH = "united_species_dataset.pkl"


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_metadata(chosen_species='human'):
    """ read metadata files from each species and concat them to one dataframe """
    if chosen_species == 'human':
        meta_data = pd.read_csv("human_length.tsv", sep="\t",
                                names=['Gene stable ID', 'transcript_id', 'transcript_start', 'transcript_end',
                                       'Chromosome'], low_memory=False).set_index('transcript_id')
        meta_data['transcript_len'] = meta_data['transcript_end'] - meta_data['transcript_start'] + 1
        return meta_data
    elif chosen_species == 'mouse':
        meta_data = pd.read_csv("mouse_length.tsv", sep="\t",
                                names=['Gene stable ID', 'transcript_id', 'transcript_start', 'transcript_end',
                                       'Chromosome']).set_index('transcript_id')
        meta_data['transcript_len'] = meta_data['transcript_end'] - meta_data['transcript_start'] + 1
        return meta_data


def load_exons(exons_path):
    exon_loaded = pd.read_pickle(exons_path)
    exon_loaded.set_index('exon_id', inplace=True)
    exon_loaded['exon_len'] = exon_loaded['exon_end'] - exon_loaded['exon_start'] + 1
    return exon_loaded


def load_data(path_to_data=DEF_DATA_PATH):
    loaded_data = pd.read_pickle(path_to_data)
    loaded_data['gene_id'] = loaded_data.index.map(lambda x: x.split('|')[0])
    loaded_data['transcript_id'] = loaded_data.index.map(lambda x: x.split('|')[2])
    loaded_data['unspliced_len'] = loaded_data['unspliced_transcript'].apply(lambda x: len(x))
    loaded_data['coding_seq_len'] = loaded_data['coding_seq'].apply(lambda x: len(x))
    loaded_data.reset_index(inplace=True)
    loaded_data.set_index('transcript_id', inplace=True)
    loaded_data = loaded_data[
        ['gene_id', 'unspliced_transcript', 'coding_seq', 'gene_id', 'unspliced_len', 'coding_seq_len', 'species']]
    return loaded_data


def find_transcripts_to_filter(dataset, metadata):
    all_transcripts = dataset.index
    common_transcripts = all_transcripts.intersection(metadata.index)

    different_len = [transcript for transcript in common_transcripts if
                     dataset.loc[transcript]['unspliced_len'] != metadata.loc[transcript]['transcript_len']]
    not_found = [transcript for transcript in all_transcripts if transcript not in common_transcripts]
    return different_len, not_found


def filter_by_chromosome(metadata, chromosome_arr, data):
    """ filter by chromosome and return only transcripts with unspliced_len == transcript_len"""
    merged_data = data.merge(metadata, left_index=True, right_index=True)
    merged_data = merged_data[merged_data['Chromosome'].isin(chromosome_arr)]
    return merged_data[merged_data['unspliced_len'] == merged_data['transcript_len']]


def get_exon_intron_lists(dataset, metadata, not_found):
    encoded_transcripts = {}
    not_same_spliced_length = []
    start_end_exons_all = {}
    start_end_introns_all = {}
    for transcript in tqdm(dataset.index):
        if transcript in different_len or transcript in not_found: continue
        exons = all_exons[all_exons['transcript_id'] == transcript].sort_values(by='exon_start', ascending=True)
        transcript_data = data.loc[transcript]
        encoded_seq = [0 for i in range(transcript_data['unspliced_len'])]
        transcript_start_genome = metadata.loc[transcript]['transcript_start']
        start_end_exons_all[transcript] = []
        start_end_introns_all[transcript] = []
        for exon, exon_data in exons.iterrows():
            start = exon_data['exon_start'] - transcript_start_genome
            end = exon_data['exon_end'] - transcript_start_genome
            # create start,end tuples for exons
            start_end_exons_all[transcript].append([start, end])
            # update encoded_seq
            for i in range(start, end + 1):
                encoded_seq[i] = 1
        ##
        # create start,end tuples for introns
        for i in range(len(start_end_exons_all[transcript]) - 1):
            start_intron = start_end_exons_all[transcript][i][1] + 1
            end_intron = start_end_exons_all[transcript][i + 1][0] - 1
            start_end_introns_all[transcript].append([start_intron, end_intron])
        ##

        # assert that sum of encoded seq == coding_seq_len
        if sum(encoded_seq) != transcript_data['coding_seq_len']: not_same_spliced_length.append(transcript)
        encoded_transcripts[transcript] = encoded_seq

    return encoded_transcripts, start_end_exons_all, start_end_introns_all


def encode_transcripts(data, metadata, all_exons):
    encoded_transcripts_dict = {}
    for transcript in tqdm(data.index):

        exons = all_exons[all_exons['transcript_id'] == transcript].sort_values(by='exon_start', ascending=True)
        transcript_data = data.loc[transcript]

        encoded_seq = [0 for i in range(transcript_data['unspliced_len'])]
        transcript_start_genome = metadata.loc[transcript]['transcript_start']

        for exon, exon_data in exons.iterrows():
            start = exon_data['exon_start'] - transcript_start_genome
            end = exon_data['exon_end'] - transcript_start_genome
            for i in range(start, end + 1):
                encoded_seq[i] = 1
        encoded_transcripts_dict[transcript] = encoded_seq

    return encoded_transcripts_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_path', type=str)
    parser.add_argument('-chrm', '--chosen_chromosomes', nargs='+', type=int)
    parser.add_argument('-train', '--train_or_test', type=str)

    args = parser.parse_args()

    # Access the parsed arguments
    data_path = args.data_path
    chosen_chromosomes = args.chosen_chromosomes
    chosen_chromosomes = list(map(str, chosen_chromosomes))
    train_or_test = args.train_or_test

    # run the pipeline
    metadata = load_metadata()
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    all_exons = load_exons("all_exons.pkl")
    data = load_data(data_path)
    print("all data loaded")

    different_len, not_found = find_transcripts_to_filter(data, metadata)
    data_filtered_by_chromosome = filter_by_chromosome(metadata, chosen_chromosomes, data)
    data_filtered_by_chromosome = data_filtered_by_chromosome[
        ~data_filtered_by_chromosome.duplicated(subset='gene_id', keep='first')]

    save_obj(not_found, f"load_for_data/not_found_{train_or_test}")
    print("not found saved to pickle")
    save_obj(different_len, f"load_for_data/different_len_{train_or_test}")
    print("different len saved to pickle")
    save_obj(data_filtered_by_chromosome, f"load_for_data/data_filtered_by_chromosome_{train_or_test}")
    print("data filtered and saved to pickle")

    encoded_transcripts = encode_transcripts(data_filtered_by_chromosome, metadata, all_exons)
    save_obj(encoded_transcripts, f"load_for_data/encoded_transcripts_{train_or_test}")
    print("encoded transcripts and saved to pickle")


