from utils import trim_long_seqs, load_fasta_file, remove_short_seqs
import random

def get_sample_from_fasta_by_max_length(max_seq_length = 256, sample_size=10000):
    # read the fasta file and remove sequences longer than max_seq_length
    fasta_file_path = 'data/uniprot_sprot.fasta'
    dataset_raw = load_fasta_file(fasta_file_path)
    dataset_preprocessed = trim_long_seqs(dataset_raw, max_seq_length)

    # get a sample of sample_size sequences
    sample = random.sample(dataset_preprocessed, sample_size)

    # write the sample to a file
    filename = 'data/sample_'+str(sample_size)+'.fasta'
    with open(filename, 'w') as f:
        for i, seq in enumerate(sample):
            f.write(f'>seq{i}\n{seq}\n')

    return sample

def get_sample_from_fasta_by_min_length(name = 'sample', min_seq_length=256, sample_size=100000):
    # read the fasta file and remove sequences shorter than min_seq_length
    fasta_file_path = 'data/uniprot_sprot.fasta'
    dataset_raw = load_fasta_file(fasta_file_path)
    dataset_preprocessed = remove_short_seqs(dataset_raw, min_seq_length = min_seq_length)

    sample = random.sample(dataset_preprocessed, sample_size)

    # write the sample to a file
    filename = 'data/sample_' + name + '.fasta'
    with open(filename, 'w') as f:
        for i, seq in enumerate(sample):
            f.write(f'>seq{i}\n{seq}\n')

    return sample

# function to get a sample of sequences from the fasta file by both max and min length
def get_sample_from_fasta_by_min_and_max_lengths(name = 'sample', min_seq_length=128, max_seq_length=384, sample_size=100000):
    # read the fasta file and remove sequences shorter than min_seq_length
    fasta_file_path = 'data/uniprot_sprot.fasta'
    dataset_raw = load_fasta_file(fasta_file_path)
    dataset_preprocessed = remove_short_seqs(dataset_raw, min_seq_length = min_seq_length)
    dataset_preprocessed = trim_long_seqs(dataset_preprocessed, max_seq_length)

    sample = random.sample(dataset_preprocessed, sample_size)

    # write the sample to a file
    if name == 'sample':
        name = 'sample' + '-' + str(len(sample)) + '_min' + str(min_seq_length) + '_max' + str(max_seq_length)
    filename = 'data/' + name + '.fasta'
    with open(filename, 'w') as f:
        for i, seq in enumerate(sample):
            f.write(f'>seq{i}\n{seq}\n')

    return sample

if __name__ == '__main__':
    #get_sample_from_fasta_by_min_length(name = '100k', min_seq_length=256, sample_size=100000)
    #get_sample_from_fasta_by_min_and_max_lengths()
    get_sample_from_fasta_by_min_and_max_lengths(name = 'sample', min_seq_length=128, max_seq_length=256, sample_size=5)