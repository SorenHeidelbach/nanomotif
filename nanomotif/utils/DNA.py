


def reverse_complement(seq: str) -> str:
    '''Returns the reverse complement of a DNA sequence'''
    assert set("ATG").issubset(["A", "G", "C", "T", "N"]), "Sequence must be a DNA sequence of A, T, G, C or N"
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
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
