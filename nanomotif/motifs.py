# %%
from typing import List, Union, Optional
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import entropy
from itertools import compress
import nanomotif.dna as dna
from nanomotif.dna import DNAsequences
from nanomotif.utils import all_lengths_equal
# %%
class SequenceEnrichment(DNAsequences):
    """
    This class provides functionality to analyse a set of equal length DNA sequences for sequence enrichment.

    Parameters
    ----------
    sequences : list
        A list of sequences for motif identification. Each sequence should be of the same length.

    Attributes
    ----------
    sequences : list
        Stores the input sequences.

    Examples
    --------
    >>> seqs = SequenceEnrichment(['ATCG', 'GCTA', 'TACG'])
    >>> seqs.sequences
    ['ATCG', 'GCTA', 'TACG']
    """
    def __init__(self, sequences: List[str]):
        assert isinstance(sequences, list), "DNA sequences must be a list"
        sequences = [seq.upper() for seq in sequences]
        super().__init__(sequences)
        self.kl_prior = np.ones((4, self.seq_length))/4

    @DNAsequences.sequences.setter
    def sequences(self, value: List[str]):
        self.sequence_alphabet = list(set("".join(value)))
        super()._check_sequences(value)
        assert all_lengths_equal(value), "All sequences must be of equal length"
        self.seq_length = len(value[0])
        self._sequences = value
    
    @property
    def kl_prior(self):
        return self._kl_prior
    
    @kl_prior.setter
    def kl_prior(self, value: np.ndarray):
        assert value.shape == (4, self.seq_length), "Prior shape does not match (should be [4, seq_length])"
        self._kl_prior = value
    
    @property
    def kl_kmer_prior(self):
        return self._kl_prior
    
    @kl_kmer_prior.setter
    def kl_kmer_prior(self, value: np.ndarray):
        assert value.shape[1] == self.seq_length, "K-mer prior shape does not match (should be [n_kmers, seq_length])"
        self._kl_prior = value

    def seq_to_int(self):
        """
        Convert stored sequences from strings of ATGC to an array of integers (1, 2, 3, 4).

        Returns
        -------
        np.ndarray
            A numpy array of stored sequences represented as integers.

        Examples
        --------
        >>> seqs = SequenceEnrichment(['ATGG', 'GGGG', 'CCGT'])
        >>> seqs.seq_to_int()
        array([[1, 2, 3, 3],
               [3, 3, 3, 3],
               [4, 4, 3, 2]])
        """
        sequences_int = []
        for i in range(len(self.sequences)):
            r = []
            for j in range(self.seq_length):
                r.append(self.base_to_int[self.sequences[i][j]])
            sequences_int.append(r)
        return np.array(sequences_int)

    def pssm(self, pseudocount=0.5):
        """
        Calculate the positional frequency of each nucleotide for all sequences.

        Parameters
        ----------
        pseudocount : float, optional
            Pseudocount to be added to each nucleotide count. Default is 0.5.

        Returns
        -------
        np.ndarray
            A numpy array of positional frequencies.

        Examples
        --------
        >>> seqs = SequenceEnrichment(['GTCG', 'GATG', 'TACG', 'TACG'])
        >>> seqs.pssm()
        array([[0.125, 0.875, 0.125, 0.125],
               [0.625, 0.375, 0.375, 0.125],
               [0.625, 0.125, 0.125, 1.125],
               [0.125, 0.125, 0.875, 0.125]])
        """
        sequences_int = self.seq_to_int()
        pos_freq = []
        for nuc in range(1, 5):
            pos_freq.append((np.sum(sequences_int == nuc, axis=0) + pseudocount)/sequences_int.shape[0])
        return np.array(pos_freq)
    
    def consensus(self):
        """
        Calculate the consensus sequence of all sequences.

        If equal frequncy is observed, the nucleotide is choosen based on the order of keys SequenceEnrichment.base_to_int.keys()

        Returns
        -------
        str
            The consensus sequence.

        Examples
        --------
        >>> seqs = SequenceEnrichment(['ATGCG', 'TTGCG', 'GCCCG', 'CCCCA'])
        >>> seqs.consensus()
        'ATGCG'
        >>> seqs.consensus()[0] == list(seqs.base_to_int.keys())[0]
        True
        """
        sequences_int = self.seq_to_int()
        consensus = []
        for position in range(self.seq_length):
            most_common_nucleotide_index = np.argmax(np.bincount(sequences_int[:, position]))
            consensus.append(self.int_to_base[most_common_nucleotide_index])
        return "".join(consensus)

    def update_kl_prior(self, prior):
        assert prior.shape == self.pssm().shape, "Prior shape does not match (should be [4, seq_length])"
        self.kl_prior = prior

    def kl_divergence(self):
        """
        Calculate positional Kullback-Liebler divergence with uniform prior.

        Returns
        -------
        np.ndarray
            A numpy array of Kullback-Leibler divergence.

        Examples
        --------
        >>> seqs = SequenceEnrichment(['GTCG', 'GATG', 'TACG', 'TACG'])
        >>> seqs.kl_divergence()
        array([0.24258597, 0.31115504, 0.31115504, 0.54930614])
        """
    
        return entropy(self.pssm(), self.kl_prior)
    
    def kl_masked_consensus(self, min_kl=0.5, mask = "N"):
        """
        Calculate the consensus sequence of all sequences with positions with low Kullback-Liebler divergence masked by ".".

        Parameters
        ----------
        min_kl : float, optional
            Minimum Kullback-Leibler divergence for a position to be considered as part of the consensus sequence. Default is 0.5.

        Returns
        -------
        str
            The consensus sequence with positions with low Kullback-Liebler divergence masked by ".".

        Examples
        --------
        >>> seqs = SequenceEnrichment(['ATCG', 'GTTA', 'TTCG', 'ATCT', 'ATCT', 'ATCT'])
        >>> seqs.kl_masked_consensus()
        'NTNN'
        """

        consensus_array = np.array([*self.consensus()])
        consensus_array[self.kl_divergence() < min_kl] = mask
        return "".join(consensus_array)

    def kl_masked_sequences(self, min_kl=0.2, mask = "N"):
        """
        Mask sequences with low Kullback-Liebler divergence with `mask`.

        Parameters
        ----------
        min_kl : float, optional
            Minimum Kullback-Leibler divergence for a position to be considered as part of the consensus sequence. Default is 0.5.

        Returns
        -------
        list
            A list of masked sequences.

        Examples
        --------
        >>> seqs = SequenceEnrichment(['ATCG', 'GTTA', 'TTCG', 'ATCT', 'ATCT', 'ATCT'])
        >>> seqs.kl_masked_sequences()
        ['ATCN', 'GTTN', 'TTCN', 'ATCN', 'ATCN', 'ATCN']
        """
        sequences_array = np.array([[*seq] for seq in self.sequences])
        sequences_array[:, self.kl_divergence() < min_kl] = mask
        return ["".join(s) for s in  sequences_array.tolist()]

    def get_motif_candidates(self, min_kl=0.2, padded=True):
        enriched_positions = self.kl_divergence() > min_kl
        enriched_bases = self.kl_prior < self.pssm()

        motif_candidates = [""]
        bases_order = np.array(list(self.int_to_base.values())[0:4])
        for i in range(enriched_bases.shape[1]):
            if bool(enriched_positions[i]):
                bases_to_add = bases_order[enriched_bases[:, i]]
                motif_candidates = [p + q for p in motif_candidates for q in bases_to_add]
            elif padded:
                motif_candidates = [p + "." for p in motif_candidates]
        
        return motif_candidates


    def cluster(self, min_cluster_size="frac", min_kl=0.01):
        """
        Cluster sequences based on Manhattan edit distance using HDBSCAN.

        Parameters
        ----------
        min_cluster_size : str or int, optional
            Minimum number of sequences required to form a cluster. If "frac", then it's 
            set to 20% of the total number of sequences or 1, whichever is larger. 
            If an integer is provided, it is used as is. Default is "frac".

        Raises
        ------
        Exception
            If there are no conserved positions found. This typically happens when min_kl is set too high.

        Side Effect
        ------------
        Updates self.clusters with the fitted HDBSCAN clustering object.

        Examples
        --------
        >>> seqs = SequenceEnrichment(['GATG', 'GATC', 'TCCA', 'TCCT'])
        >>> HDBSCAN = seqs.cluster(min_cluster_size = 2)
        >>> type(seqs.clusters)
        <class 'dict'>
        >>> seqs.clusters[0].sequences
        ['GATG', 'GATC']
        >>> seqs.clusters[1].sequences
        ['TCCA', 'TCCT']
        >>> type(HDBSCAN)
        <class 'hdbscan.hdbscan_.HDBSCAN'>
        """
        # Clusters sequences
        sequences_int = self.seq_to_int()
        if min_cluster_size == "frac":
            min_cluster_size = int(max(sequences_int.shape[0] * 0.20, 2))

        clust = hdbscan.HDBSCAN(metric ="hamming", min_cluster_size = int(min_cluster_size))
        evaluated_positions = sequences_int[:, self.kl_divergence() > min_kl]
        if evaluated_positions.shape[1] == 0:
            raise Exception("No conserved positions found. Try lowering min_kl")
        clust.fit(evaluated_positions)

        # Store clusters
        clusters = {i: SequenceEnrichment(list(compress(self.sequences, clust.labels_==i))) for i in set(clust.labels_)}
        self.clusters = clusters
        return clust


    def dreme(self, kmer_size=4):
        positives = self.sequences
        negatives = [dna.generate_random_dna_sequence(self.seq_length) for _ in range(len(self.sequences))]
        return dna.dreme_naive(positives, negatives, kmer_size)

    def plot_enrichment(self, ax=None, min_kl=0.5, center_x_axis=True):
        """
        Method to plot the positional conservation of DNA sequences in the object.

        Parameters:
        ax (matplotlib.axes.Axes, optional): An existing Axes object to draw the plot onto, 
                                            default is None in which case a new figure will be created.
        min_kl (float, optional): The minimum value for Kullback-Leibler divergence, default is 0.5.

        Returns:
        None

        This method generates a plot showing the conservation of sequences in the object.
        It plots the positional frequencies of each nucleotide, as well as the Kullback-Leibler divergence.
        """
        
        # If no Axes object provided, create a new figure and axes
        if ax is None:
            _, ax = plt.subplots()

        # Calculate positional frequency
        positional_freq = self.pssm(pseudocount=0)

        # Plot positional frequencies for each nucleotide
        for i in range(1,5):
            ax.plot(positional_freq[i-1], label=f"{self.int_to_base[i]}", linewidth=2, c=cm.cubehelix(i/5))

        # Plot Kullback-Leibler divergence
        ax.plot(self.kl_divergence(), "--", label="KL", color="black")
        ax.axhline(y=min_kl, ls="--", color="grey", label="KL lim")

        # Add text annotations for masked consensus sequence
        for base, pos in zip(self.kl_masked_consensus(min_kl=min_kl), range(int(self.seq_length))):
            ax.text(pos, -0.01, base, color="black", fontsize=10, va="top", ha="center")

        # Configure x-axis labels and ticks
        if center_x_axis:
            n_labs = min(4, (self.seq_length-1)//2) 
            window = (self.seq_length-1)//2
            x_axis_step = max(window//n_labs, 1)
            labs_positive = np.arange(0, x_axis_step * n_labs + 1, x_axis_step)
            tick_label = np.concatenate((-labs_positive, labs_positive))
            tick_position = tick_label + window
            ax.set_xticks(tick_position)
            ax.set_xticklabels(tick_label)
            ax.set_xlabel("Relative Position")
        else:
            ax.set_xticks(np.arange(0, self.seq_length, 1))
            ax.set_xlabel("Position")
        # Set title and labels
        ax.set_title(f"Number of sequences: {len(self.sequences)}")
        ax.set_ylabel("Frequency/KL Divergence")

        return ax
    
    def plot_enrichment_with_prior(self, min_kl=0.5, center_x_axis=True):

        plot, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        self.plot_enrichment(min_kl = min_kl, ax=ax[0], center_x_axis=center_x_axis)

        for i in range(self.kl_prior.shape[0]):
            ax[1].plot(self.kl_prior[i, :] + (i/120), label=f"{self.int_to_base[i+1]}", linewidth=2, c=cm.cubehelix((i+1)/5))


        if center_x_axis:
            n_labs = min(4, (self.seq_length-1)//2) 
            window = (self.seq_length-1)//2
            x_axis_step = max(window//n_labs, 1)
            labs_positive = np.arange(0, x_axis_step * n_labs + 1, x_axis_step)
            tick_label = np.concatenate((-labs_positive, labs_positive))
            tick_position = tick_label + window
            ax[1].set_xticks(tick_position)
            ax[1].set_xticklabels(tick_label)
            ax[1].set_xlabel("Relative Position")
        else:
            ax[1].set_xticks(np.arange(0, self.seq_length, 1))
            ax[1].set_xlabel("Position")

        ax[1].set_title("Prior")

        ax[0].legend()
        plot.tight_layout()

        return plot, ax
    
    def plot_kmer_graph(self, kmer_size = 3, stride = 1, min_connection=0, kl_threshold=None):
        if kl_threshold is not None:
            seqs = self.kl_masked_sequences(min_kl=kl_threshold)
        else:
            seqs = self.sequences
        kmer_graph = dna.kmer_graph(seqs, kmer_size=kmer_size, stride=stride)
        dna.plot_kmer_graph(kmer_graph, min_connection=min_connection)



# %%
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    dna.generate_random_dna_sequence(10)
    enriched_seqs_CTGAAG = [ 
        dna.generate_random_dna_sequence(5) +
        dna.generate_random_dna_sequence(1, alphabet=["C"]*97 + ["T", "A", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["T"]*97 + ["A", "C", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["G"]*97 + ["T", "C", "A"]) +
        dna.generate_random_dna_sequence(1, alphabet=["A"]*97 + ["T", "C", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["A"]*97 + ["T", "C", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["G"]*97 + ["T", "C", "A"]) +
        dna.generate_random_dna_sequence(5)
        for _ in range(500)
    ]
    enriched_seqs_CTTCAG = [ 
        dna.generate_random_dna_sequence(5) +
        dna.generate_random_dna_sequence(1, alphabet=["C"]*97 + ["T", "A", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["T"]*97 + ["A", "C", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["T"]*97 + ["G", "C", "A"]) +
        dna.generate_random_dna_sequence(1, alphabet=["C"]*97 + ["T", "A", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["A"]*97 + ["T", "C", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["G"]*97 + ["T", "C", "A"]) +
        dna.generate_random_dna_sequence(5)
        for _ in range(300)
    ]
    enriched_seqs_CAAT = [ 
        dna.generate_random_dna_sequence(7) +
        dna.generate_random_dna_sequence(1, alphabet=["C"]*97 + ["T", "A", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["A"]*97 + ["T", "C", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["A"]*97 + ["G", "C", "T"]) +
        dna.generate_random_dna_sequence(1, alphabet=["T"]*97 + ["C", "A", "G"]) +
        dna.generate_random_dna_sequence(5)
        for _ in range(200)
    ]
    two_motifs = SequenceEnrichment(
        enriched_seqs_CTGAAG + 
        enriched_seqs_CTTCAG + 
        enriched_seqs_CAAT
        )

    two_motifs.plot_kmer_graph(kmer_size=4, stride=1, min_connection=0)
    # %%
