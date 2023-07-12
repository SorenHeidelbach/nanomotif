import numpy as np
import hdbscan
import copy
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib import cm
from scipy.stats import entropy
from itertools import compress

from nanomotif.utils.dna import dreme_naive, generate_random_dna_sequence 


class SequenceEnrichment:
    """
    This class provides functionality to analyse a set of equal length DNA sequences for sequence enrichment.
    It supports various functionalities, such as calculating positional frequency of each nucleotide, 
    extracting consensus sequence, sequence clustering, positional Kullback-Leibler divergence and more.

    Parameters
    ----------
    sequences : list
        A list of sequences for motif identification. Each sequence should be of the same length.

    Attributes
    ----------
    sequences : list
        Stores the input sequences.
    seq_length : int
        Length of a single sequence. All sequences should be of this length.
    nuc_to_int : dict
        Mapping of nucleotide to integer.
    int_to_nuc : dict
        Mapping of integer to nucleotide.
    clusters : HDBSCAN
        HDBSCAN clustering object of sequences based on manhattan distance.

    Raises
    ------
    AssertionError
        If the sequences are not of equal length or there are fewer than 2 sequences.

    Examples
    --------
    >>> seqs = SequenceEnrichment(['ATCG', 'GCTA', 'TACG'])
    >>> seqs.sequences
    ['ATCG', 'GCTA', 'TACG']
    """
    def __init__(self, sequences: list):
        assert len(sequences) >= 2, "SequenceEnrichment much have at least 2 sequences"
        self.sequences = sequences
        self.seq_length = len(sequences[0])
        assert  all([self.seq_length == len(i) for i in self.sequences ]) , "Sequences are not all the same length"

        self.nuc_to_int = {"A":1, "T":2, "G":3, "C":4, "N":5, "R":6, "Y":7, "S":8, "W":9, "K":10, "M":11, "B":12, "D":13, "H":14, "V":15}
        self.int_to_nuc = {i: n for n, i in self.nuc_to_int.items()}

        # KL prior is initialised as a uniform prior
        self.kl_prior = np.full_like(self.pssm(), 0.25)

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
                r.append(self.nuc_to_int[self.sequences[i][j]])
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

        If equal frequncy is observed, the nucleotide is choosen based on the order of keys SequenceEnrichment.nuc_to_int.keys()

        Returns
        -------
        str
            The consensus sequence.

        Examples
        --------
        >>> seqs = SequenceEnrichment(['ATGCG', 'TTGCG', 'GCCCG', 'CCCCA'])
        >>> seqs.consensus()
        'ATGCG'
        >>> seqs.consensus()[0] == list(seqs.nuc_to_int.keys())[0]
        True
        """
        sequences_int = self.seq_to_int()
        consensus = []
        for position in range(self.seq_length):
            most_common_nucleotide_index = np.argmax(np.bincount(sequences_int[:, position]))
            consensus.append(self.int_to_nuc[most_common_nucleotide_index])
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
    
    def kl_masked_consensus(self, min_kl=0.5):
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
        '.T..'
        """

        consensus_array = np.array([*self.consensus()])
        consensus_array[self.kl_divergence() < min_kl] = "."
        return "".join(consensus_array)

    def kl_masked_sequences(self, min_kl=0.5):
        """
        Mask sequences with low Kullback-Liebler divergence with ".".

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
        ['ATCG', '.TT.', 'TTCG', 'ATCT', 'ATCT', 'ATCT']
        """
        sequences_array = np.array([[*seq] for seq in self.sequences])
        sequences_array[:, self.kl_divergence() < min_kl] = "."
        return sequences_array

    def get_motif_candidates(self, min_kl=0.2, padded=True):
        enriched_positions = self.kl_divergence() > min_kl
        enriched_bases = self.kl_prior < self.pssm()

        motif_candidates = [""]
        bases_order = np.array(list(self.int_to_nuc.values())[0:4])
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
        >>> type(seqs.clusters[0])
        <class 'nanomotif.motifs.SequenceEnrichment'>
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
        negatives = [generate_random_dna_sequence(self.seq_length) for _ in range(len(self.sequences))]
        return dreme_naive(positives, negatives, kmer_size)

    def plot_enrichment(self, ax=None, min_kl=0.5):
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
            fig, ax = plt.subplots()

        # Calculate positional frequency
        positional_freq = self.pssm(pseudocount=0)

        # Plot positional frequencies for each nucleotide
        for i in range(1,5):
            ax.plot(positional_freq[i-1], label=f"{self.int_to_nuc[i]}", linewidth=2, c=cm.cubehelix(i/5))

        # Plot Kullback-Leibler divergence
        ax.plot(self.kl_divergence(), "--", label="KL", color="black")
        ax.axhline(y=min_kl, ls="--", color="grey", label="KL lim")

        # Add text annotations for masked consensus sequence
        for nuc, pos in zip(self.kl_masked_consensus(min_kl=min_kl), range(int(self.seq_length))):
            ax.text(pos, -0.01, nuc, color="black", fontsize=10, va="top", ha="center")

        # Configure x-axis labels and ticks
        n_labs = 4
        window = self.seq_length//2
        x_axis_step = window//n_labs
        labs_positive = np.arange(0, x_axis_step * n_labs + 1, x_axis_step)
        tick_label = np.concatenate((-labs_positive, labs_positive))
        tick_position = tick_label + window
        ax.set_xticks(tick_position)
        ax.set_xticklabels(tick_label)

        # Set title and labels
        ax.set_title(f"Number of sequences: {len(self.sequences)}")
        ax.set_xlabel("Relative Position")
        ax.set_ylabel("Frequency/KL Divergence")

        return ax
    
    def plot_enrichment_with_prior(self, min_kl=0.5):

        plot, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        self.plot_enrichment(min_kl = min_kl, ax=ax[0])

        for i in range(self.kl_prior.shape[0]):
            ax[1].plot(self.kl_prior[i, :] + (i/120), label=f"{self.int_to_nuc[i+1]}", linewidth=2, c=cm.cubehelix((i+1)/5))

        # Configure x-axis labels and ticks
        n_labs = 4
        window = self.seq_length//2
        x_axis_step = window//n_labs
        labs_positive = np.arange(0, x_axis_step * n_labs + 1, x_axis_step)
        tick_label = np.concatenate((-labs_positive, labs_positive))
        tick_position = tick_label + window
        ax[1].set_xticks(tick_position)
        ax[1].set_xticklabels(tick_label)
        ax[1].set_xlabel("Relative Position")
        ax[1].set_title("Prior")

        ax[0].legend()
        plot.tight_layout()

        return plot, ax



def plot_dna_sequences(sequences, cm_palette="Pastel2"):
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
    """
    
    # Create SequenceEnrichment and append consensus sequence
    m = SequenceEnrichment(copy.copy(sequences))
    m.sequences.append(m.consensus())

    n_seqs, n_bases = len(m.sequences), len(m.sequences[0])

    # Generate labels for sequences
    y_tick_labels = [f"Seq {i}" for i in range(n_seqs-1)] + ["Consensus"]
    
    # Define mapping from nucleotide bases to integers
    base_to_int = {"A": 1, "G": 2, "C": 3, "T": 4}

    # Create 2D array of sequences
    sequences_array = np.array([base_to_int[n] for seq in m.sequences for n in seq])
    sequences_array = sequences_array.reshape(n_seqs, n_bases)
    
    # Calculate font size
    font_size = 15

    # Create plot
    fig, ax = plt.subplots(figsize=(2 + n_bases*.28, 1 + n_seqs*.28));
    ax.imshow(sequences_array, cmap=cm.get_cmap(cm_palette));

    # Set labels for y axis
    ax.set_yticks(np.arange(n_seqs));
    ax.set_yticklabels(y_tick_labels, fontsize=font_size, fontweight="bold");

    # Loop over data dimensions and create text annotations
    for i in range(n_seqs):
        for j in range(n_bases):
            ax.text(j, i, m.sequences[i][j], ha="center", va="center", color="black", 
                    fontsize=font_size);
    
    # Draw line to separate consensus from sequences
    ax.axhline(y=n_seqs-1.5, color="black", linewidth=1);
    
    # Adjust plot layout and display
    fig.tight_layout()

    return ax




if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)