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

    base_to_int = {"A":1, "T":2, "G":3, "C":4, "N":5, "R":6, "Y":7, "S":8, "W":9, "K":10, "M":11, "B":12, "D":13, "H":14, "V":15}
    int_to_base = {i: n for n, i in base_to_int.items()}

    def __init__(self, sequences):
        assert isinstance(sequences, list), "DNA sequences must be a list"
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
        assert all_equal(lengths), "All sequences must be of equal length"

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
               [4, 4, 3, 2]])
        >>> DNAsequences(['ATGG', 'GGGGG', 'CCGT']).sequences_int()
        array([list([1, 2, 3, 3]), list([3, 3, 3, 3, 3]), list([4, 4, 3, 2])],
              dtype=object)
        """
        return np.array(list(map(
            lambda x: list(map(
                lambda y: DNAsequences.base_to_int[y], 
                x
            )), 
            self
        )))
    
    def trimmed_sequences(self) -> list:
        """
        Trim sequences to the same length as the shortest sequence.

        Returns
        -------
        list
            List of trimmed sequences.

        Examples
        --------
        >>> DNAsequences(['ATGG', 'GGGGG', 'CCGT']).trimmed_sequences()
        ['ATGG', 'GGGG', 'CCGT']
        >>> DNAsequences(['A', 'ATG', 'CCGTT']).trimmed_sequences()
        ['A', 'A', 'C']
        """
        lengths = [len(s) for s in self]
        min_length = min(lengths)
        seqs = [seq[:min_length] for seq in self]
        return seqs

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
        >>> DNAsequences(['ATGG', 'GGGG', 'CCGT']).padded_sequences()
        ['ATGG', 'GGGG', 'CCGT']
        >>> DNAsequences(['ATGG', 'GGGG', 'CCGTT']).padded_sequences(pad = "A")
        ['ATGGA', 'GGGGA', 'CCGTT']
        """
        assert pad in self.iupac, f"pad must be one of {self.iupac}"
        lengths = [len(s) for s in self]
        max_length = max(lengths)
        seqs = [seq + pad * (max_length - len(seq)) for seq in self]
        return seqs

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
        if not all_lengths_equal(self):
            if trim:
                seqs = self.trimmed_sequences()
            else:
                seqs = self.padded_sequences(pad=pad)
        else:
            seqs = self.sequences
        
        return seqs

    def pssm(self, pseudocount=0, trim=True, pad="N", only_canonical=True):
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

    def consensus(self, trim=True, pad="N"):
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
        seqs = self._get_equal_length_sequences(trim=trim, pad=pad)

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

    def kmers(self, kmer_size: int) -> list:
        """
        Generate all k-mers from a given sequence.

        Parameters
        ----------
        kmer_size : int
            The length of the k-mers.

        Returns
        -------
        list
            A list of k-mers. Each k-mer is a string. The list is empty if kmer_size is greater than the length of the sequence.

        Examples
        --------
        >>> DNAsequences(["", "A", "AT", "ATGA"]).kmers(2)
        [[], [], ['AT'], ['AT', 'TG', 'GA']]
        """
        assert isinstance(kmer_size, int), "kmer_size must be an integer"
        assert kmer_size > 0, "kmer_size must be greater than 0"
        return [[seq[i:i+kmer_size] for i in range(len(seq) - kmer_size + 1)] for seq in self]
    
    def sample_sequences(self, n):
        """
        Sample sequences from the DNAsequences object.

        Returns
        -------
        list
            A list of sampled sequences.

        Examples
        --------
        >>> random.seed(0)
        >>> DNAsequences(["A", "TA", "CTA", "GTCA"]).sample_sequences(3)
        ['GTCA', 'GTCA', 'TA']
        """
        return random.choices(self.sequences, k=n)



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


def get_kmer_graph(sequences, kmer_size = 6, stride = 2, position_aware = True):
    """
    Get a kmer graph from DNAsequences.

    Args:
        sequences (List[str]): A list of sequences to build kmer graph.
        kmer_size (int, optional): The length of kmers to consider. Defaults to 6.
        stride (int, optional): The step size to take when considering subsequences. Defaults to 2.
        position_aware (bool, optional): Whether to group by position or not. Defaults to True.

    Returns:
        DataFrame: A DataFrame representing the kmer graph with columns: 'from', 'to', 'count', ('position').
            from: The kmer at the start of the edge.
            to: The kmer at the end of the edge.
            count: The number of times the edge appears in the sequences (by position if position_aware = True).
            (position: The position of the kmer in the sequence.)

    Examples:
    >>> get_kmer_graph(["GATC", "GATG", "GATA", "GATG"], kmer_size = 2, stride = 1, position_aware = True)
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

    >>> get_kmer_graph(["GATC", "GATG", "GATA", "GATG"], kmer_size = 3, stride = 1, position_aware = False)
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
    assert min(len(seq) for seq in sequences) >= kmer_size + stride, "All sequences must be at least as long as kmer_size+stride"
    # Initialize a DataFrame with sequences
    kmer_graph = pl.DataFrame({"seq": sequences})

    # Define an apply function for getting kmers
    def get_kmers_and_positions(seq):
        kmers = DNAsequences([seq]).kmers(kmer_size)[0]
        kmers_from = kmers[::stride][:-1]
        kmers_to = kmers[::stride][1:]
        positions = list(range(0, (len(seq) - kmer_size) - stride + 1, stride))
        
        return {"from": kmers_from, "to": kmers_to, "position": positions}
    
    # Apply the function to the sequence column
    kmer_graph = kmer_graph.with_columns(pl.col("seq").apply(get_kmers_and_positions).alias("order"))
    kmer_graph = kmer_graph.unnest("order").explode("from", "to", "position")

    # Group by according to the position_aware flag
    if position_aware:
        kmer_graph = kmer_graph.groupby(["from", "to", "position"]).agg(pl.count().alias("count")) 
    else:
        kmer_graph = kmer_graph.groupby(["from", "to"]).agg(pl.count().alias("count"))
    
    return kmer_graph.sort("from", "to", "count")

def plot_kmer_graph(kmer_graph, min_connection = 10, ax=None, y_axis = "mean_connections"):
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


def all_equal(iterator):
    """
    Checks whether all elements in an iterable are equal.

    The function will return True even if the iterable is empty. It works with any iterable that supports 
    equality comparison, including strings, lists, and tuples.

    Args:
        iterator (Iterable): An iterable object.

    Returns:
        bool: True if all elements are equal or if iterable is empty, False otherwise.

    Examples:
    >>> all_equal([1, 1, 1])
    True
    >>> all_equal('aaa')
    True
    >>> all_equal([])
    True
    >>> all_equal([1, 2])
    False
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def all_lengths_equal(iterator):
    """
    Checks whether the lengths of all elements in an iterable are equal.

    The function will return True even if the iterable is empty. It requires that the elements of the iterable
    also be iterable, such as strings, lists, and tuples.

    Args:
        iterator (Iterable): An iterable object containing other iterable elements.

    Returns:
        bool: True if all lengths are equal or if iterable is empty, False otherwise.

    Examples:
    >>> all_lengths_equal(['abc', 'def', 'ghi'])
    True
    >>> all_lengths_equal([[1, 2, 3], [4, 5, 6]])
    True
    >>> all_lengths_equal([])
    True
    >>> all_lengths_equal(['abc', 'de'])
    False
    """
    iterator = iter(iterator)
    try:
        first = len(next(iterator))
    except StopIteration:
        return True
    return all(first == len(x) for x in iterator)


if __name__ == "__main__":
    import doctest
    doctest.testmod()


