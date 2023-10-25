import pandas as pd
from tqdm import tqdm
import json
import sys
import random
import pickle
import argparse

DEF_DATA_PATH = "../data/united_species_dataset.pkl"


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_metadata(chosen_species='human'):
    """ read metadata files from each species and concat them to one dataframe """
    if chosen_species == 'human':
        meta_data = pd.read_csv("../data/human_length.tsv", sep="\t",
                                names=['Gene stable ID', 'transcript_id', 'transcript_start', 'transcript_end',
                                       'Chromosome'], low_memory=False).set_index('transcript_id')
        meta_data['transcript_len'] = meta_data['transcript_end'] - meta_data['transcript_start'] + 1
        return meta_data
    elif chosen_species == 'mouse':
        meta_data = pd.read_csv("../data/mouse_length.tsv", sep="\t",
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
        transcript_data = dataset.loc[transcript]
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


def create_random_subsets_by_length(encoded_dict, length, transcript, filtered_data):
    """ return dict of sub_seq - sub_encoded of length l"""
    encoded_transcript = encoded_dict[transcript]
    n_cut = len(encoded_transcript) // length

    # create subsets and their [start,end] in the original seq to find the true sequence
    subsets = []
    subsets_positions = {}
    for i in range(n_cut):
        random_start = random.randint(0, len(encoded_transcript) - length)
        subset = encoded_transcript[random_start:random_start + length]
        subsets_positions[i] = [random_start, random_start + length]
        subsets.append(subset)

    # select a subset of 70% only introns and 30% mixed + only exons
    only_exons = {idx: subset for idx, subset in enumerate(subsets) if subset.count(0) == 0}
    only_introns = {idx: subset for idx, subset in enumerate(subsets) if subset.count(1) == 0}
    mixed = {idx: subset for idx, subset in enumerate(subsets) if subset.count(0) != 0 and subset.count(1) != 0}
    n_introns = round(n_cut * 0.7)
    n_mixed = min(round(n_cut * 0.3), len(mixed))
    selected_introns = [key for key in only_introns.keys()][:n_introns]
    selected_mixed = [key for key in mixed.keys()][:n_mixed]

    # final set of subsets encoded and their [start,end] in the original seq
    final_subsets = [only_introns[idx] for idx in selected_introns] + [mixed[idx] for idx in selected_mixed] + list(
        only_exons.values())
    final_subsets_positions = [subsets_positions[i] for i in selected_introns] + [subsets_positions[i] for i in
                                                                                  selected_mixed] + [
                                  subsets_positions[i] for i in only_exons]

    seq_to_subsets_dict = {}
    full_seq = filtered_data.loc[transcript]['unspliced_transcript']
    for i, sub in enumerate(final_subsets):
        start = final_subsets_positions[i][0]
        end = final_subsets_positions[i][1]
        seq = full_seq[start:end]
        seq_to_subsets_dict[seq] = sub

    return seq_to_subsets_dict


# Functions to create data for the model
def get_num_samples(transcript_length, exons_list, current_exon):
    exons_length = sum(end - start + 1 for start, end in exons_list)
    curr_exon_length = current_exon[1] - current_exon[0] + 1
    num_exon_samples = int(transcript_length * (curr_exon_length / exons_length) / 100)
    return num_exon_samples


def create_data(dataset, encoded_transcripts_dict, start_end_exons_all,seq_input_len, to_sample_len):
    """
    :param dataset: data_for_model of transcripts containes the 'unslipced_transcript'
    :param encoded_transcripts_dict: dictionary of transcript labels
    :param start_end_exons_all: dictionary of exons ranges per transcript
    :param seq_input_len: size of the sequence input given to the model
    :param to_sample_len: size of range to sample from around each exon
    :return: data_for_model = {transcript: [(seq, label)...], ...}
    """
    if seq_input_len > to_sample_len:
        raise ValueError("seq_input_len should be less than or equal to to_sample_len")

    middle_nuc_pos = int((seq_input_len-1)/2)  # position of the middle nucleotide in the input sequence

    # data_for_model = {transcript: [(seq, label)]}
    data_for_model = {}

    for transcript in dataset.index:
        transcript_seq = dataset['unspliced_transcript'][transcript]
        transcript_labels = encoded_transcripts_dict[transcript]
        if len(transcript_seq) > seq_input_len:  # transcript is long enough
            data_for_model[transcript] = []
            # define range to sample around the current exon
            for i, exon in enumerate(start_end_exons_all[transcript]):
                if exon[0] < to_sample_len:  # too close to the beginning of the transcript sequence
                    start = 0
                else:
                    start = exon[0] - to_sample_len
                if exon[1] + to_sample_len > len(transcript_seq)-1: # to close to the end of the transcript sequence
                    end = len(transcript_seq)-1
                else:
                    end =  exon[1] + to_sample_len
                range_to_sample = (start, end)

                # sample around the current exon
                num_exon_samples = get_num_samples(len(transcript_seq), start_end_exons_all[transcript], exon)
                for _ in range(num_exon_samples):
                    start_position = random.randint(range_to_sample[0], range_to_sample[1] - seq_input_len)
                    end_position = start_position + seq_input_len
                    seq = transcript_seq[start_position:end_position]
                    seq_labels = transcript_labels[start_position:end_position]
                    label = seq_labels[middle_nuc_pos]

                    data_for_model[transcript].append((seq, label))

    return data_for_model

def transform_data(original_data):
    """
    change data from {transcript: [(seq, label)]} format to [(seq, label, transcript)] format
    :param original_data: original_data --> {transcript: [(seq, label)]}
    :return: transformed_data --> [(seq, label, transcript)]
    """
    transformed_data = []
    for transcript, samples in original_data.items():
        for seq, label in samples:
            transformed_data.append((seq, label, transcript))
    return transformed_data


def balance_data(original_data):
    all_positive_examples = [(seq, label, transcript) for (seq, label, transcript) in original_data if label == 1]
    all_negative_examples = [(seq, label, transcript) for (seq, label, transcript) in original_data if label == 0]

    # Find the number of samples from the minority category
    num_samples_per_category = min(len(all_positive_examples), len(all_negative_examples))

    # Randomly sample an equal number of positive and negative examples
    sampled_data = (
            random.sample(all_positive_examples, num_samples_per_category) +
            random.sample(all_negative_examples, num_samples_per_category)
    )

    # Shuffle the sampled data to mix positive and negative examples
    random.shuffle(sampled_data)
    return sampled_data


def model_data_to_json(model_data, data_info, json_path):
    json_data = []
    for x, y, transcript in tqdm(model_data):
        gene_id = data_info.loc[transcript]['gene_id']
        if type(gene_id) is not str:
            gene_id = gene_id.values[0]
        json_dict = {'unspliced_transcript': x,
                     'class': y,
                     'gene_id': gene_id}
        json_data.append(json_dict)
    with open(json_path, "w") as final:
        json.dump(json_data, final)
    return json_data


def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--chosen_length', type=int)
    parser.add_argument('-train', '--train_or_test', type=str)

    args = parser.parse_args()

    # Access the parsed arguments
    chosen_length = args.chosen_length
    train_or_test = args.train_or_test

    # run the pipeline
    all_exons = load_exons("../data/all_exons.pkl")
    metadata = load_metadata()
    metadata = metadata[~metadata.index.duplicated(keep='first')]

    not_found = load_obj(f"../load_for_data/not_found_{train_or_test}")
    different_len = load_obj(f"../load_for_data/different_len_{train_or_test}")
    data_filtered_by_chromosome = load_obj(f"../load_for_data/data_filtered_by_chromosome_{train_or_test}")
    print("data loaded")

    _, start_end_exons_all, start_end_introns_all = get_exon_intron_lists(data_filtered_by_chromosome, metadata,
                                                                          not_found)
    # TODO change to load encoded transcripts
    encoded_transcripts = load_obj(f"../load_for_data/encoded_transcripts_{train_or_test}")
    print("loaded encoded transcripts")

    X = create_data(data_filtered_by_chromosome, encoded_transcripts, start_end_exons_all, seq_input_len=chosen_length,
                    to_sample_len=chosen_length)
    transformed_X = transform_data(X)
    balanced_X = balance_data(transformed_X)
    print("created data for model")

    json_data = model_data_to_json(balanced_X, data_filtered_by_chromosome,
                                   f"../updated_data_by_len/human_data_{chosen_length}_{train_or_test}.json")
    print("converted to json file")
