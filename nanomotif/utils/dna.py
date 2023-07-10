import random
import numpy as np
from collections import Counter

def reverse_complement(seq: str) -> str:
    '''Returns the reverse complement of a DNA sequence'''
    complement = {
        "A": "T", 
        "T": "A", 
        "G": "C", 
        "C": "G", 
        "N": "N", 
        "R": "Y", 
        "Y": "R", 
        "S": "S", 
        "W": "W", 
        "K": "M", 
        "M": "K", 
        "B": "V", 
        "D": "H", 
        "H": "D", 
        "V": "B"
    }
    assert set(seq).issubset(list(complement.keys())), "Sequence must be a DNA sequence of A, T, G, C or N"

    return "".join(complement[base] for base in reversed(seq))

def generate_random_dna_sequence(length):
    nucleotides = ['A', 'T', 'C', 'G']
    sequence = ''.join(random.choice(nucleotides) for _ in range(length))
    return sequence


def generate_dna_sequence_with_random_padding(seq: str, length: int):
    nucleotides = ['A', 'T', 'C', 'G']
    length = length - len(seq)
    sequence = ''.join(random.choice(nucleotides) for _ in range(length))
    sequence = seq + sequence
    return sequence

def generate_kmers(k: int):
    nucleotides = ['A', 'T', 'C', 'G']
    kmers = []
    for i in range(4**k):
        kmer = ""
        for j in range(k):
            kmer += nucleotides[i % 4]
            i = i // 4
        kmers.append(kmer)
    return kmers

def sample_seq_at_letter(seq, n, size, letter):
    """
    Sample n sequences of length 2*size+1 from seq at positions where letter is in the middle.

    Parameters
    ----------
    seq : str
        Sequence from which to sample.
    n : int
        Number of sequences to sample.
    size : int
        Size of the sequences to sample.
    letter : str
        Letter to sample at.

    Returns
    -------
    sequences : list
        List of sequences of length 2*size+1 sampled from seq at positions where letter is in the middle.
    """
    seq_length = len(seq)

    positions = [i for i, ltr in enumerate(seq) if ltr in letter and i-size >= 0 and i+size+1 < seq_length]
    if len(positions) == 0:
        return []
    else:
        postions_sampled = random.choices(positions, k=n)
        return [seq[i-size:i+size+1] for i in postions_sampled]

def get_gc_content(seq: str) -> float:
    '''Returns the GC content of a sequence'''
    return (seq.count("G") + seq.count("C")) / len(seq)



def get_kmers(sequence: str, kmer_size: int) -> list:
    """
    Generate all k-mers from a given sequence.

    Parameters
    ----------
    sequence : str
        The DNA sequence from which to generate k-mers.
    kmer_size : int
        The length of the k-mers.

    Returns
    -------
    list
        A list of all k-mers of given size that can be generated from the input sequence.

    Examples
    --------
    >>> get_kmers('ATCGAT', 2)
    ['AT', 'TC', 'CG', 'GA', 'AT']
    """
    return [sequence[i:i+kmer_size] for i in range(len(sequence) - kmer_size + 1)]



def calculate_enrichment(positive_counts: dict, negative_counts: dict) -> dict:
    """
    Calculate the enrichment of k-mers in the positive sequences compared to negative sequences.

    Parameters
    ----------
    positive_counts : dict
        A dictionary containing the counts of each k-mer in the positive sequences.
    negative_counts : dict
        A dictionary containing the counts of each k-mer in the negative sequences.

    Returns
    -------
    dict
        A dictionary containing the log2 enrichment of each k-mer in the positive sequences.

    Examples
    --------
    >>> positive_counts = {'AT': 10, 'TC': 10, 'CG': 5}
    >>> negative_counts = {'AT': 5, 'TC': 5, 'CG': 5}
    >>> enrichments = calculate_enrichment(positive_counts, negative_counts)
    >>> enrichments['AT']  # Enrichment of 'AT' in the positive sequences.
    0.26303440583379406
    """
    total_positive_counts = sum(positive_counts.values())
    total_negative_counts = sum(negative_counts.values())

    enrichments = {}
    for kmer, pos_count in positive_counts.items():
        positive_freq = pos_count / total_positive_counts
        negative_freq = negative_counts.get(kmer, 0) / total_negative_counts
        enrichments[kmer] = np.inf if negative_freq == 0 else np.log2(positive_freq / negative_freq)
    return enrichments


def dreme_naive(positive_sequences: list, negative_sequences: list, kmer_size: int = 4) -> dict:
    """
    A simplified version of the DREME algorithm to find enriched k-mers in positive sequences.

    Parameters
    ----------
    positive_sequences : list
        A list of DNA sequences considered positive.
    negative_sequences : list, optional
        A list of DNA sequences considered negative. If set to "random", generates random sequences. Default is "random".
    kmer_size : int, optional
        The length of the k-mers. Default is 6.

    Returns
    -------
    dict
        A dictionary containing the enrichment of each k-mer in the positive sequences.

    Examples
    --------
    >>> SequenceEnrichment = SequenceEnrichment(['ATCG', 'GCTA', 'TACG', 'AGCT'])
    >>> positive_sequences = ['ATCG', 'GCTA', 'TACG', 'AGCT']
    >>> negative_sequences = ['CGTA', 'TACG', 'GATC', 'CGAT']
    >>> enrichments = dreme_naive(positive_sequences, negative_sequences, kmer_size=2)
    >>> isinstance(enrichments, dict)
    True
    """
    # Count k-mers in positive sequences
    positive_counts = Counter(kmer for sequence in positive_sequences for kmer in get_kmers(sequence, kmer_size))

    # Count k-mers in negative sequences
    negative_counts = Counter(kmer for sequence in negative_sequences for kmer in get_kmers(sequence, kmer_size))

    enrichments = calculate_enrichment(positive_counts, negative_counts)

    return enrichments
