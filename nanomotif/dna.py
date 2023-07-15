from typing import List
import random
import numpy as np
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib as mpl
import editdistance
import nanomotif.utils as utils
from matplotlib import cm
from collections import Counter

class DNAsequences:
    bases = ["A", "T", "C", "G"]
    iupac = ["A", "T", "C", "G", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N"]
    iupac_dict = {
        "A": ["A"], "T": ["T"], "C": ["C"], "G": ["G"],
        "R": ["A", "G"], "Y": ["C", "T"], 
        "S": ["G", "C"], "W": ["A", "T"], 
        "K": ["G", "T"], "M": ["A", "C"],
        "B": ["C", "G", "T"],
        "D": ["A", "G", "T"],
        "H": ["A", "C", "T"],
        "V": ["A", "C", "G"],
        "N": ["A", "T", "C", "G"]
    }
    complement = {
        "A": "T", "T": "A", "G": "C", "C": "G", 
        "N": "N", "R": "Y", "Y": "R", "S": "S", 
        "W": "W", "K": "M", "M": "K", "B": "V", 
        "D": "H", "H": "D", "V": "B"
    }

    base_to_int = {"A":1, "T":2, "G":3, "C":4, "N":5, "R":6, "Y":7, "S":8, "W":9, "K":10, "M":11, "B":12, "D":13, "H":14, "V":15}
    int_to_base = {i: n for n, i in base_to_int.items()}

    def __init__(self, sequences):
        self._check_sequences(sequences)
        sequences = [seq.upper() for seq in sequences]
        self.sequences = sequences
    
    @property
    def sequences(self):
        return self._sequences

    @sequences.setter
    def sequences(self, value):
        self.sequence_alphabet = list(set("".join(value)))
        self._check_sequences(value)
        self._sequences = value

    def __getitem__(self, key):
        return self.sequences[key]

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)
    
    def __repr__(self):
        return f"DNAsequences() | Number of sequences: {len(self)} | Unique Bases: {self.sequence_alphabet}"

    def __add__(self, other):
        if isinstance(other, DNAsequences):
            return DNAsequences(self.sequences + other.sequences)
        elif isinstance(other, list):
            return DNAsequences(self.sequences + other)
        else:
            return NotImplemented

    def __mul__(self, other):
        try:
            return DNAsequences(self.sequences * other)
        except:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, DNAsequences):
            return self.sequences == other.sequences
        elif isinstance(other, list):
            return self.sequences == other
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, DNAsequences):
            return self.sequences != other.sequences
        elif isinstance(other, list):
            return self.sequences != other
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, int):
            return len(self) < other
        elif isinstance(other, (DNAsequences, list, set, tuple)):
            return len(self) < len(other)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, int):
            return len(self) <= other
        elif isinstance(other, (DNAsequences, list, set, tuple)):
            return len(self) <= len(other)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, int):
            return len(self) > other
        elif isinstance(other, (DNAsequences, list, set, tuple)):
            return len(self) > len(other)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, int):
            return len(self) >= other
        elif isinstance(other, (DNAsequences, list, set, tuple)):
            return len(self) >= len(other)
        else:
            return NotImplemented

    def __contains__(self, item):
        return item in self.sequences    

    def __copy__(self):
        return DNAsequences(self.sequences.copy())

    def __deepcopy__(self):
        return DNAsequences(self.sequences.copy())

    def _check_sequences(self, sequences):
        assert isinstance(sequences, list), "DNA sequences must be a list"
        assert len(sequences) > 0, "DNA sequences must not be empty"
        assert all(isinstance(seq, str) for seq in sequences), "DNA sequences must be a list of strings"
        assert all(len(seq) > 0 for seq in sequences), "DNA sequences must not be empty strings"
        assert all(letter in self.iupac for letter in list(set("".join(sequences)))), f"DNA sequences must be a nucleotide sequence of {''.join(self.iupac)}"
    
    def reverse_complement(self) -> list:
        '''
        Returns the reverse complement of sequences

        Returns
        -------
        list
            List of reverse complemented sequences

        Examples
        --------
        >>> DNAsequences(["ATCG", "GCTA", "TACG", "AGCT"]).reverse_complement()
        ['CGAT', 'TAGC', 'CGTA', 'AGCT']
        '''
        return ["".join([self.complement[base] for base in reversed(seq)]) for seq in self]

    def gc(self) -> list:
        '''
        Returns the GC content of a sequences

        Returns
        -------
        list of floats
            GC content of sequences

        Examples
        --------
        >>> DNAsequences(["ATCG", "GCTA", "TACA", "AGCT"]).gc()
        [0.5, 0.5, 0.25, 0.5]
        '''
        return [(seq.count("G") + seq.count("C")) / len(seq) for seq in self]

    def levenshtein_distances(self) -> np.ndarray:
        """
        All vs. all levenshtein distances between sequences in the DNAsequences object.

        Returns
        -------
        np.ndarray
            A matrix of edit distances between all sequences in the DNAsequences object.

        Examples
        --------
        >>> DNAsequences(["ATCG", "GCTA", "TACT", "AGCT"]).levenshtein_distances()
        array([[0, 4, 3, 2],
               [4, 0, 3, 2],
               [3, 3, 0, 2],
               [2, 2, 2, 0]], dtype=int16)
        """
        lengths = [len(s) for s in self]
        assert utils.all_equal(lengths), "All sequences must be of equal length"

        n = len(self)
        distances = np.empty(shape = (n, n), dtype=np.int16)
        for i in range(0, n):
            for j in range(i, n):
                d = editdistance.eval(self[i], self[j])
                distances[i, j] = d
                distances[j, i] = d
        return distances

    def sequences_int(self):
        """
        Convert stored sequences from strings of ATGC to an array of integers (1, 2, 3, 4).

        Returns
        -------
        np.ndarray
            A numpy array of stored sequences represented as integers.

        Examples
        --------
        >>> DNAsequences(['ATGG', 'GGGG', 'CCGT']).sequences_int()
        array([[1, 2, 3, 3],
               [3, 3, 3, 3],
               [4, 4, 3, 2]], dtype=int16)
        >>> DNAsequences(['ATGG', 'GGGGG', 'CCGT']).sequences_int()
        array([list([1, 2, 3, 3]), list([3, 3, 3, 3, 3]), list([4, 4, 3, 2])],
              dtype=object)
        """
        if utils.all_lengths_equal(self):
            return np.array(list(map(
                lambda x: list(map(
                    lambda y: DNAsequences.base_to_int[y], 
                    x
                )), 
                self
            )), dtype=np.int16)
        else:
            return np.array(list(map(
                lambda x: list(map(
                    lambda y: DNAsequences.base_to_int[y], 
                    x
                )), 
                self
            )), dtype=object)
    
    def max_length(self) -> int:
        """
        Get the length of the longest sequence.

        Returns
        -------
        int
            Length of the longest sequence.

        Examples
        --------
        >>> DNAsequences(['ATGG', 'GGGGG', 'CCGT']).max_length()
        5
        >>> DNAsequences(['A', 'ATG', 'CCGTT']).max_length()
        5
        """
        return max([len(s) for s in self])

    def min_length(self) -> int:
        """
        Get the length of the shortest sequence.

        Returns
        -------
        int
            Length of the shortest sequence.

        Examples
        --------
        >>> DNAsequences(['ATGG', 'GGGGG', 'CCGT']).min_length()
        4
        >>> DNAsequences(['A', 'ATG', 'CCGTT']).min_length()
        1
        """
        return min([len(s) for s in self])

    def trimmed_sequences(self) -> list:
        """
        Trim sequences to the same length as the shortest sequence.

        Returns
        -------
        list
            List of trimmed sequences.

        Examples
        --------
        >>> DNAsequences(['ATGG', 'GGGGG', 'CCGT']).trimmed_sequences().sequences
        ['ATGG', 'GGGG', 'CCGT']
        >>> DNAsequences(['A', 'ATG', 'CCGTT']).trimmed_sequences().sequences
        ['A', 'A', 'C']
        """
        lengths = [len(s) for s in self]
        min_length = min(lengths)
        seqs = [seq[:min_length] for seq in self]
        return DNAsequences(seqs)

    def padded_sequences(self, pad: str = "N") -> list:
        """
        Pad sequences to the same length with a given character.

        Parameters
        ----------
        pad : str, optional
            Character to pad sequences with. Default is "N".
            Must be one of IUPAC nucleotides: A, T, C, G, R, Y, S, W, K, M, B, D, H, V, N
        
        Returns
        -------
        list
            List of padded sequences.

        Examples
        --------
        >>> DNAsequences(['ATGG', 'GGGG', 'CCGT']).padded_sequences().sequences
        ['ATGG', 'GGGG', 'CCGT']
        >>> DNAsequences(['ATGG', 'GGGG', 'CCGTT']).padded_sequences(pad = "A").sequences
        ['ATGGA', 'GGGGA', 'CCGTT']
        """
        assert pad in self.iupac, f"pad must be one of {self.iupac}"
        lengths = [len(s) for s in self]
        max_length = max(lengths)
        seqs = [seq + pad * (max_length - len(seq)) for seq in self]
        return DNAsequences(seqs)

    def _get_equal_length_sequences(self, trim: bool = True, pad: str = "N") -> list:
        """
        Convert all sequences to the same length.

        Args:
            trim (bool, optional): Whether to trim the end of seuqences to match shortest sequence. Defaults to True.
            pad (str, optional): Character to pad sequences with. Default is "N".
                Must be one of IUPAC nucleotides: A, T, C, G, R, Y, S, W, K, M, B, D, H, V, N

        Returns:
            list: List of sequences of equal length.
        """
        if not utils.all_lengths_equal(self):
            if trim:
                seqs = self.trimmed_sequences()
            else:
                seqs = self.padded_sequences(pad=pad)
        else:
            seqs = self.sequences
        
        return seqs

    def pssm(self, pseudocount=0, trim: bool = True, pad: str = "N", only_canonical: bool = True):
        """
        Calculate the positional frequency of each nucleotide for all sequences.

        Parameters
        ----------
        pseudocount : float, optional
            Pseudocount to be added to each nucleotide count. Default is 0.5.
        
        trim : bool, optional
            Whether to trim the end of seuqences to match shortest sequence. Default is True.
            Pads with N to match longest sequence if False.

        Returns
        -------
        np.ndarray
            A numpy array of positional frequencies.

        Examples
        --------
        >>> DNAsequences(["A", "TA", "CTA", "GTCA"]).pssm()
        array([[0.25],
               [0.25],
               [0.25],
               [0.25]])
        >>> DNAsequences(["A", "TA", "CTA", "GTCA"]).pssm(trim=False)
        array([[0.25, 0.25, 0.25, 0.25],
               [0.25, 0.5 , 0.  , 0.  ],
               [0.25, 0.  , 0.25, 0.  ],
               [0.25, 0.  , 0.  , 0.  ]])
        >>> DNAsequences(["W", "TK", "CTA", "GBCA"]).pssm(only_canonical=False, trim=False)
        array([[0.  , 0.  , 0.25, 0.25],
               [0.25, 0.25, 0.  , 0.  ],
               [0.25, 0.  , 0.25, 0.  ],
               [0.25, 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  ],
               [0.25, 0.  , 0.  , 0.  ],
               [0.  , 0.25, 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.25, 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.25, 0.5 , 0.75]])
        """
        seqs = self._get_equal_length_sequences(trim=trim, pad=pad)
        
        if only_canonical:
            valid_base = self.bases
        else:
            valid_base = self.iupac

        seq_length = len(seqs[0])
        n_seqs = len(seqs)

        pssm = []
        for nuc in valid_base:
            pssm_nuc = []
            for i in range(seq_length):
                count = 0
                for seq in seqs:
                    if seq[i] == nuc:
                        count += 1
                pssm_nuc.append((count + pseudocount)/n_seqs)
            pssm.append(pssm_nuc)
        return np.array(pssm)

    def consensus(self, trim: bool = True, pading: str = "N"):
        """
        Calculate the consensus sequence of all sequences.

        If equal frequncy is observed, the nucleotide is choosen based on the order of keys in the iupac_dict.

        Returns
        -------
        str
            The consensus sequence.

        Examples
        --------
        >>> DNAsequences(['ATGCG', 'TTGCG', 'GCCCG', 'CCCCA']).consensus()
        'ATGCG'
        """
        # Ensure all sequences are of equal length
        seqs = self._get_equal_length_sequences(trim=trim, pad=pading)

        # Initialize consensus sequence as an empty string
        consensus_sequence = ''

        # Loop over each position in the sequences
        for position in range(len(seqs[0])):

            # Create a Counter for each nucleotide at the current position across all sequences
            nucleotide_counts = Counter(seq[position] for seq in seqs)

            # Determine the most common nucleotide at the current position
            most_common_nucleotide = nucleotide_counts.most_common(1)[0][0]

            # Append the most common nucleotide to the consensus sequence
            consensus_sequence += most_common_nucleotide
        return consensus_sequence

    def kmers(self, kmer_size: int, sequence_index: int = None) -> list:
        """
        Generate all k-mers from sequences.

        Parameters
        ----------
        kmer_size : int
            The length of the k-mers.
        sequence_index : int, optional
            The index of the sequence to generate k-mers from. Default is None, which generates k-mers from all sequences.

        Returns
        -------
        list
            A list of k-mers. Each k-mer is a string. The list is empty if kmer_size is greater than the length of the sequence.

        Examples
        --------
        >>> DNAsequences(["T", "A", "AT", "ATGA"]).kmers(kmer_size = 2)
        [[], [], ['AT'], ['AT', 'TG', 'GA']]
        """
        assert isinstance(kmer_size, int), "kmer_size must be an integer"
        assert kmer_size > 0, "kmer_size must be greater than 0"
        if sequence_index is not None:
            assert isinstance(sequence_index, int), "sequence_index must be an integer"
            assert sequence_index >= 0, "sequence_index must be greater than or equal to 0"
            assert sequence_index < len(self), "sequence_index must be less than the number of sequences"
            seqs = [self[sequence_index]]
        else:
            seqs = self.sequences
        return [get_kmers_in_sequence(seq, kmer_size=kmer_size) for seq in seqs]
    
    def sample_sequences(self, n: int):
        """
        Sample sequences from the DNAsequences object.

        Returns
        -------
        DNAsequences
            A DNAsequences object containing the sampled sequences.

        Examples
        --------
        >>> random.seed(0)
        >>> seqs = DNAsequences(["A", "TA", "CTA", "GTCA"]).sample_sequences(3)
        >>> type(seqs)
        <class 'nanomotif.dna.DNAsequences'>
        >>> seqs.sequences
        ['GTCA', 'TA', 'A']
        """
        return DNAsequences(random.sample(self.sequences, k=n))
    
    def sample_kmers(self, kmer_size: int, n_kmers: int=None, n_seqs: int=None):
        """
        Sample k-mers from all sequences.
        Multiple samling is not allowed, meaning all sequences and kmers are only samples at most once.

        Returns
        -------
        kmer_size : int
            The length of the k-mers.
        n_kmers : int
            The number of k-mers to sample. Must be greater than 0. If greater than the number of kmers in the sequences, all kmers are returned for a sequence.
        n_seqs : int
            The number of sequences to sample from.

        Examples
        --------
        >>> random.seed(0)
        >>> DNAsequences(["A", "TA", "CTA", "GTCGTAGTA"]).sample_kmers(kmer_size=2, n_kmers=2)
        [[], ['TA'], ['TA', 'CT'], ['GT', 'CG']]
        """
        if n_seqs is None:
            seqs = self.sequences
        else:
            assert isinstance(n_seqs, int), "n_seqs must be an integer"
            assert n_seqs > 0, "n_seqs must be greater than 0"
            if len(self) > n_seqs:
                seqs = self.sample_sequences(n_seqs).sequences
            else:
                seqs = self.sequences
        all_kmers = DNAsequences(seqs).kmers(kmer_size)

        if n_kmers is not None:
            assert isinstance(n_kmers, int), "n_kmers must be an integer"
            assert n_kmers > 0, "n_kmers must be greater than 0"
        
        sampled_kmers = []
        for k in all_kmers:
            if n_kmers is None:
                sampled_kmers.append(k)
            else:
                if len(k) < n_kmers:
                    sampled_kmers.append(k)
                else:
                    sampled_kmers.append(random.sample(k, k=n_kmers))

        return sampled_kmers

def generate_random_dna_sequence(length: int, alphabet: List[str] = ['A', 'T', 'C', 'G']):
    """
    Generate a random DNA sequence from the specified alphabet.
    """
    sequence = ''.join(random.choices(alphabet, k=length))
    return sequence

def pad_dna_seq(seq: str, target_length: int, alphabet: list[str] = ['A', 'T', 'C', 'G'], where: str = "sides"):
    """
    Pad a DNA sequence with random nucleotides.
    """
    assert  target_length >= len(seq), "target_length must be greater or equal to the length of seq"
    pad_length = target_length - len(seq)

    if where == "left":
        sequence = seq + generate_random_dna_sequence(pad_length, alphabet=alphabet)
    elif where == "right":
        sequence = generate_random_dna_sequence(pad_length, alphabet=alphabet) + seq
    elif where == "sides":
        pad_length_right = pad_length // 2
        pad_length_left = pad_length - pad_length_right
        sequence = generate_random_dna_sequence(pad_length_left, alphabet=alphabet) + seq + generate_random_dna_sequence(pad_length_right, alphabet=alphabet)
    return sequence

def pad_dna_seq_n(seq: str, target_length: int, where: str = "sides"):
    """
    Pad a DNA sequence with Ns.    
    """
    return pad_dna_seq(seq, target_length, alphabet=['N'], where=where)

def generate_all_possible_kmers(k: int, alphabet: list[str] = ['A', 'T', 'C', 'G']):
    """
    Generate k-mers of length k.
    """
    unique_bases = len(alphabet)
    kmers = []
    for i in range(unique_bases**k):
        kmer = ""
        for j in range(k):
            kmer += alphabet[i % unique_bases]
            i = i // unique_bases
        kmers.append(kmer)
    return kmers

def convert_ints_to_bases(ints: list[int], conversion_dict: dict):
    seq = ""
    for i in ints:
        seq = seq + conversion_dict[i]
    return seq


def sample_seq_at_letter(seq: str, n: int, size: int, letter: str):
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


def dreme_naive(positive_sequences: list[str], negative_sequences: list[str], kmer_size: int = 4) -> dict:
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
    >>> positive_sequences = ['ATCG', 'GCTA', 'TACG', 'AGCT']
    >>> negative_sequences = ['CGTA', 'TACG', 'GATC', 'CGAT']
    >>> enrichments = dreme_naive(positive_sequences, negative_sequences, kmer_size=2)
    >>> isinstance(enrichments, dict)
    True
    """
    # Count k-mers in positive sequences
    positive_counts = Counter(
        utils.flatten_list(DNAsequences(positive_sequences).kmers(kmer_size))
    )

    # Count k-mers in negative sequences
    negative_counts = Counter(
        utils.flatten_list(DNAsequences(negative_sequences).kmers(kmer_size))
    )
    enrichments = calculate_enrichment(positive_counts, negative_counts)

    return enrichments


def get_kmers_in_sequence(sequence: str, kmer_size: int = 3):
    """
    Generate all k-mers from a sequence.

    Args:
        sequence (str): The sequence to generate k-mers from.
        kmer_size (int, optional): The length of the k-mers. Defaults to 3.

    Returns:
        List[str]: A list of k-mers. Each k-mer is a string. The list is empty if kmer_size is greater than the length of the sequence.

    Examples:
    >>> get_kmers_in_sequence("ATCG", kmer_size = 2)
    ['AT', 'TC', 'CG']
    >>> get_kmers_in_sequence("ATCG", kmer_size = 3)
    ['ATC', 'TCG']
    >>> get_kmers_in_sequence("ATCG", kmer_size = 4)
    ['ATCG']
    """
    assert isinstance(sequence, str), "sequence must be a string"
    assert isinstance(kmer_size, int), "kmer_size must be an integer"
    assert kmer_size > 0, "kmer_size must be greater than 0"
    if kmer_size > len(sequence):
        return []
    else:
        return [sequence[i:i+kmer_size] for i in range(len(sequence) - kmer_size + 1)]


def kmer_graph(sequences: List[str], kmer_size: int = 3, stride: int = 1, position_aware: bool = True):
    """
    Identifies connections between kmers in a set of sequences to get number of transitions between K-mers.

    Args:
        sequences (List[str]): 
            A list of sequences to build kmer graph.
        kmer_size (int, optional): 
            The length of kmers to consider. Defaults to 3.
        stride (int, optional): 
            The step size to take when considering subsequences. Defaults to 1.
        position_aware (bool, optional): 
            Whether to group in context of position of the sequences. Defaults to True.

    Returns:
        DataFrame: A DataFrame representing the kmer graph with columns: 'from', 'to', 'count', ('position').
            from: The kmer at the start of the edge.
            to: The kmer at the end of the edge.
            count: The number of times the edge appears in the sequences (by position if position_aware = True).
            (position: The position of the start of the kmer in the sequence. 0 index)

    Examples:
    >>> kmer_graph(["GATC", "GATG", "GATA", "GATG"], kmer_size = 2, stride = 1, position_aware = True)
    shape: (4, 4)
    ┌──────┬─────┬──────────┬───────┐
    │ from ┆ to  ┆ position ┆ count │
    │ ---  ┆ --- ┆ ---      ┆ ---   │
    │ str  ┆ str ┆ i64      ┆ u32   │
    ╞══════╪═════╪══════════╪═══════╡
    │ AT   ┆ TA  ┆ 1        ┆ 1     │
    │ AT   ┆ TC  ┆ 1        ┆ 1     │
    │ AT   ┆ TG  ┆ 1        ┆ 2     │
    │ GA   ┆ AT  ┆ 0        ┆ 4     │
    └──────┴─────┴──────────┴───────┘

    >>> kmer_graph(["GATC", "GATG", "GATA", "GATG"], kmer_size = 3, stride = 1, position_aware = False)
    shape: (3, 3)
    ┌──────┬─────┬───────┐
    │ from ┆ to  ┆ count │
    │ ---  ┆ --- ┆ ---   │
    │ str  ┆ str ┆ u32   │
    ╞══════╪═════╪═══════╡
    │ GAT  ┆ ATA ┆ 1     │
    │ GAT  ┆ ATC ┆ 1     │
    │ GAT  ┆ ATG ┆ 2     │
    └──────┴─────┴───────┘

    """
    assert isinstance(sequences, list), "sequences must be a list"
    assert len(sequences) > 0, "sequences must not be empty"
    assert isinstance(kmer_size, int), "kmer_size must be an integer"
    assert kmer_size > 0, "kmer_size must be greater than 0"
    assert isinstance(stride, int), "stride must be an integer"
    assert isinstance(position_aware, bool), "position_aware must be a boolean"
    assert min(len(seq) for seq in sequences) >= kmer_size + stride, "All sequences must be at least as long as kmer_size+stride"
    
    seqs = DNAsequences(sequences)
    all_kmers = seqs.kmers(kmer_size)

    # Get from and to kmers
    kmers_from = [k[::stride][:-1] for k in all_kmers]
    kmers_to = [k[::stride][1:] for k in all_kmers]
    positions = [i*stride for kmer in kmers_from for i in range(len(kmer))]

    kmer_graph = pl.DataFrame({
        "from": utils.flatten_list(kmers_from), 
        "to": utils.flatten_list(kmers_to), 
        "position": positions
    })

    # Group by according to the position_aware flag
    if position_aware:
        kmer_graph = kmer_graph.groupby(["from", "to", "position"]).agg(pl.count().alias("count")).sort("from", "to", "count", "position")
    else:
        kmer_graph = kmer_graph.groupby(["from", "to"]).agg(pl.count().alias("count")).sort("from", "to", "count")
    
    return kmer_graph


def plot_kmer_graph(kmer_graph: pl.DataFrame, min_connection: int = 10, ax: plt.Axes = None, y_axis: str = "mean_connections"):
    """
    Plots a K-mer graph. Nodes in the graph represent kmers and edges represent connections between kmers. 
    The width of the edges is proportional to the number of connections between kmers. The y-coordinate of
    each node is proportional to the number of connections/mean connections to that node.
    The x-coordinate of each node is the position of the kmer in the sequence.

    Args:
        kmer_graph (DataFrame): A DataFrame representing a graph with columns 'from', 'to', 'count', 'position'. 
            'from' and 'to' represent kmers, 'count' represents the number of connections, 'position' represents 
            the position of kmers.
        min_connection (int, optional): The minimum number of connections to include in the graph. Defaults to 10.
        ax (AxesSubplot, optional): Matplotlib Axes object to draw the plot onto, otherwise uses the current Axes. 
            Defaults to None.
        y_axis (str, optional): Method of calculating the y-coordinate for each node. Can be either 'mean_connections' 
            (the mean of the count of connections) or 'connections' (the total count of connections). Defaults to 
            'mean_connections'.

    Raises:
        ValueError: If `y_axis` is neither 'mean_connections' nor 'connections'.
        AssertionError: If `kmer_graph` does not have columns 'from', 'to', 'count', 'position'.
    
    Returns:
        None

    This function will not return anything but will display the K-mer graph using matplotlib.
    """
    # Check if the kmer_graph contains all necessary columns
    required_columns = ['from', 'to', 'count', 'position']
    for column in required_columns:
        assert column in kmer_graph.columns, f"kmer_graph must have a column named '{column}'"
        
    assert y_axis in ["mean_connections", "connections"], "y_axis must be 'mean_connections' or 'connections'"
    assert min_connection >= 0, "min_connection must be greater than or equal to 0"
    assert kmer_graph.shape[0] > 0, "kmer_graph must not be empty"

    # Process the kmer_graph dataframe
    kmer_graph = kmer_graph.with_columns([
            (pl.col("from") + "_" + pl.col("position").cast(pl.Utf8)),
            (pl.col("to") + "_" + pl.col("position").add(1).cast(pl.Utf8))
        ]).filter(pl.col("count") > min_connection)

    # Convert dataframe to NetworkX DiGraph
    G = nx.from_pandas_edgelist(kmer_graph, "from", "to", ["count", "position"], create_using=nx.DiGraph)

    # Calculate position of each node
    pos = {}
    for node in G.nodes(data=True):
        filter_condition = (pl.col("from") == node[0]) | (pl.col("to") == node[0])
        if y_axis == "mean_connections":
            yval = kmer_graph.filter(filter_condition).mean().get_column("count")[0]
        else:
            yval = kmer_graph.filter(filter_condition).sum().get_column("count")[0]
        pos[node[0]] = (int(node[0].split("_")[1]), yval)

    # Calculate edge widths
    counts = [edge[2]["count"] for edge in G.edges(data=True)]
    line_width = [count / max(counts) * 10 for count in counts]

    # If no Axes object provided, create a new one
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 8))

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=False, alpha = 1, width = line_width,
        edge_cmap=mpl.colormaps["cividis_r"], edge_color = counts)

    # Draw labels
    labels = {node: node.split("_")[0] for node in pos}
    nx.draw_networkx_labels(G, pos, ax=ax, 
                            bbox=dict(boxstyle="square", fc="w", ec="k"),
                            font_color="black", font_size=10, font_family="monospace",
                            labels=labels)

    # Set axis ticks and labels
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("K-mer start position, relative to methylation site")
    ax.set_ylabel("Mean connections" if y_axis == "mean_connections" else "Total Connections")

    # Set color range for edge colors
    ax.collections[0].set_clim(0, None)
    return ax


def plot_dna_sequences(sequences: List[str], cm_palette: str = "Pastel2"):
    """
    Function to plot DNA sequences with a consensus sequence.
    
    Parameters:
    sequences (list): A list of DNA sequences to be plotted.
    cm_palette (str): A color palette to use for the plot, default is "Pastel2".

    Returns:
    None
    
    The function will plot DNA sequences in a form of heatmap and add a consensus sequence at the bottom. 
    The x-axis represents the position in the sequence, and the y-axis represents individual sequences and the consensus.

    >>> sequences = ['ATGCGAC', 'ATTCGAC', 'ATGCGAT', 'ATGCGAC']
    >>> plot_dna_sequences(sequences)
    <Axes: >
    """
    
    # Create SequenceEnrichment and append consensus sequence
    seqs = DNAsequences(sequences)
    seqs = seqs.padded_sequences()
    seqs = seqs + [seqs.consensus()]
    n_seqs, n_bases = len(seqs), seqs.max_length()
    
    # Generate labels for sequences
    y_tick_labels = [f"Seq {i}" for i in range(n_seqs-1)] + ["Consensus"]

    # Create 2D array of sequences
    sequences_array = seqs.sequences_int()
    
    # Calculate font size
    font_size = 15

    # Create plot
    fig, ax = plt.subplots(figsize=(2 + n_bases*.28, 1 + n_seqs*.28));
    ax.imshow(sequences_array, cmap=getattr(cm, cm_palette));

    # Set labels for y axis
    ax.set_yticks(np.arange(n_seqs));
    ax.set_yticklabels(y_tick_labels, fontsize=font_size, fontweight="bold");

    # Loop over data dimensions and create text annotations
    for i in range(n_seqs):
        for j in range(n_bases):
            ax.text(j, i, seqs.sequences[i][j], ha="center", va="center", color="black", 
                    fontsize=font_size);
    
    # Draw line to separate consensus from sequences
    ax.axhline(y=n_seqs-1.5, color="black", linewidth=1);
    
    # Adjust plot layout and display
    fig.tight_layout()

    return ax


# %%
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    a = DNAsequences(["ATGCGAC", "ATTCGAC", "ATGCGAT", "ATGCGAC"])
#%%