import random
import numpy as np
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import editdistance
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
    def __init__(self, sequences):
        sequences = [seq.upper() for seq in sequences]
        self.sequences = sequences
    
    @property
    def sequences(self):
        return self._sequences

    @sequences.setter
    def sequences(self, value):
        assert isinstance(value, list), "DNA sequences must be a list"
        assert all(isinstance(seq, str) for seq in value), "DNA sequences must be a list of strings"
        self.sequence_alphabet = list(set("".join(value)))
        assert all(letter in self.iupac for letter in self.sequence_alphabet), f"DNA sequences must be a nucleotide sequence of {''.join(self.iupac)}"
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

    def reverse_complement(self) -> list:
        '''
        Returns the reverse complement of a sequences

        Returns
        -------
        list
            List of reverse complemented sequences

        Examples
        --------
        >>> DNAsequences(["ATCG", "GCTA", "TACG", "AGCT"]).reverse_complement()
        ['CGAT', 'TAGC', 'CGTA', 'AGCT']
        '''
        return ["".join([self.complement[base] for base in reversed(seq)]) for seq in self.sequences]

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
        return [(seq.count("G") + seq.count("C")) / len(seq) for seq in self.sequences]

    def edit_distances(self) -> np.ndarray:
        """
        Calculate the edit distance between all sequences in the DNAsequences object.

        Returns
        -------
        np.ndarray
            A matrix of edit distances between all sequences in the DNAsequences object.

        Examples
        --------
        >>> DNAsequences(["ATCG", "GCTA", "TACT", "AGCT"]).edit_distances()
        array([[0., 4., 3., 2.],
               [4., 0., 3., 2.],
               [3., 3., 0., 2.],
               [2., 2., 2., 0.]])
        """
        n = len(self)
        dists = np.empty(shape = (n, n))
        for i in range(0, n):
            for j in range(0, n):
                dists[i, j] = editdistance.eval(self[i], self[j])
        return dists


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

def convert_ints_to_bases(ints: list[int], conversion_dict: dict):
    seq = ""
    for i in ints:
        seq = seq + conversion_dict[i]
    return seq

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

def edit_distance(sequences: list) -> np.ndarray:
    dists = np.empty(shape = (len(sequences), len(sequences)))
    for i in range(0, len(sequences)):
        for j in range(0, len(sequences)):
            dists[i, j] = editdistance.eval(sequences[i], sequences[j])
    return dists

def get_kmer_graph(sequences, kmer_size = 6, stride = 2, position_aware = True):
    kmer_graph = pl.DataFrame({"seq":sequences}).with_columns(
      pl.col("seq").apply(
        lambda seq: {
          "from":get_kmers(seq, kmer_size)[::stride][:-1], 
          "to": get_kmers(seq, kmer_size)[::stride][1:], 
          "position": range(0, (len(sequences[0]) - kmer_size) - stride + 1, stride)
        }
      ).alias("order")
    ).unnest("order").explode("from", "to", "position")

    if position_aware:
        kmer_graph = kmer_graph \
        .groupby(["from", "to", "position"]).agg(pl.count().alias("count")) 
    else:
        kmer_graph = kmer_graph \
        .groupby(["from", "to"]).agg(pl.count().alias("count"))
    return kmer_graph

def plot_kmer_graph(kmer_graph, min_connection = 10, ax=None, y_axis = "mean_connections"):
    kmer_graph = kmer_graph.with_columns([
            (pl.col("from") + "_" + pl.col("position").cast(pl.Utf8)),
            (pl.col("to") + "_" + pl.col("position").add(1).cast(pl.Utf8))
        ]).filter(pl.col("count") > min_connection)
    G = nx.from_pandas_edgelist(kmer_graph, "from", "to", ["count", "position"], create_using=nx.DiGraph)
    pos = {}
    for i in G.nodes(data=True):
        if y_axis == "mean_connections":
            yval = kmer_graph.filter((pl.col("from") == i[0]) | (pl.col("to") == i[0])).mean().get_column("count")[0]
        elif y_axis == "connections":
            yval = kmer_graph.filter((pl.col("from") == i[0]) | (pl.col("to") == i[0])).sum().get_column("count")[0]
        else:
            raise ValueError("y_axis must be 'mean_connections' or 'connections'")
        pos[i[0]] = (int(i[0].split("_")[1]), yval)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))
    
    counts = [i[2]["count"] for i in G.edges(data=True)]
    line_width = [i/max(counts) * 10 for i in counts]
    
    nx.draw_networkx_edges(
        G, pos, ax=ax, 
        arrows=False, alpha = 1, width = line_width, 
        edge_cmap=mpl.colormaps["cividis_r"], edge_color = counts
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, 
        bbox = dict(boxstyle="square", fc="w", ec="k"),
        font_color = "black", font_size=10, font_family="monospace",
        labels = {i: i.split("_")[0] for i in pos}
    )
    # Show axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    n_labs = 4
    window = 25//2
    x_axis_step = window//n_labs
    labs_positive = np.arange(0, x_axis_step * n_labs + 1, x_axis_step)
    tick_label = np.concatenate((-labs_positive, labs_positive))
    tick_position = tick_label + window
    ax.set_xticks(tick_position)
    ax.set_xticklabels(tick_label)
    ax.set_xlabel("K-mer start position, relative to methylation site")
    if y_axis == "mean_connections":
        ax.set_ylabel("Mean connections")
    elif y_axis == "connections":
        ax.set_ylabel("Total Connections")
    ax.collections[0].set_clim(0, None)

if __name__ == "__main__":
    import doctest
    doctest.testmod()