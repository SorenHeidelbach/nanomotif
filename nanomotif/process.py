from nanomotif.data_loader import DataLoader
from nanomotif.reference import Reference
from nanomotif.motifs import SequenceEnrichment
from nanomotif.utils.dna import sample_seq_at_letter, edit_distance
import numpy as np
import warnings
import copy
import polars as pl
import seaborn as sns


class ContigProcessor():
    def __init__(self, pileup_file, ref_file, window_size: int=5, min_fraction_mod: float=80, modified: str="a|m", min_score: int=5, backup_prefilt: bool=True):
        """
        Constructor of the ContigProcessor class.

        :param pileup_file: Path to the pileup file
        :type pileup_file: str
        :param ref_file: Path to the reference file
        :type ref_file: str
        :param window_size: Size of the window for sequence context
        :type window_size: int, optional
        :param min_fraction_mod: Minimum fraction of modified reads
        :type min_fraction_mod: float, optional
        :param modified: Regex of modified base to keep
        :type modified: str, optional
        :param min_score: Minimum score
        :type min_score: int, optional
        :param backup_prefilt: Backup raw pileup data
        :type backup_prefilt: bool, optional
        """
        # Input validations
        assert isinstance(window_size, int), "window_size must be an integer"
        assert window_size > 0, "window_size must be positive"

        self.ref = Reference(ref_file, trim_names=True)
        self.raw_pileup = DataLoader(pileup_file).pileup 
        self.filter_pileup(min_fraction_mod, modified, min_score)
        self.pileup = self.add_ref_sequence_at_modification(window_size) 
        self.sequence_enrichments_stranded = self.instantiate_sequence_enrichments()
        self.sequence_enrichments = self.combine_sequence_enrichments_on_strand()
        self.update_kl_priors()
        self.kl_prior_type = "uniform"
        self.window_size = window_size

        # Instance validation
        assert isinstance(self.pileup, pl.DataFrame), "pileup must be a DataFrame"
        assert isinstance(self.ref, Reference), "ref must be a Reference object"

    def update_motif_identifiers(self):
        """
        Function to update motif identiers, reflecting the current pileup date.
        """

        self.sequence_enrichments_stranded = self.instantiate_sequence_enrichments()
        self.sequence_enrichments = self.combine_sequence_enrichments_on_strand()

    def add_ref_sequence_at_modification(self, window_size: int=5):
        """
        Function to add reference sequence at the modification site.

        :param window_size: Size of the window for sequence context
        :type window_size: int, optional
        :returns: Updated pileup data with sequence context added
        :rtype: DataFrame
        """

        # Input validations
        assert isinstance(window_size, int), "window_size must be an integer"
        assert window_size > 0, "window_size must be positive"
        assert isinstance(self.pileup, pl.DataFrame), "pileup must be a DataFrame"
        assert isinstance(self.ref, Reference), "ref must be a Reference object"

        pileup = self.pileup

        try:
            pileup = pileup.with_columns(
                pl.struct(["ref", "start"])
                    .apply(lambda x: self.ref.get_surrounding_sequence(x["ref"], x["start"], window_size))
                    .alias("sequence")
            )
            self.window_size = window_size
        except Exception as e:
            raise ValueError(f"An error occurred while fetching the sequence: {str(e)}")

        return pileup
    
    def filter_pileup(self, min_fraction_mod: float=70, modified: str="a|m", min_score: int=5):
        """
        Filter the pileup data.

        :param min_fraction_mod: Minimum fraction of modified reads
        :type min_fraction_mod: float
        :param modified: Regex of modified base to keep
        :type modified: str
        :param min_score: Minimum score
        :type min_score: int

        :returns: Filtered pileup data
        :rtype: DataFrame
        """

        try:
            pileup_filt = self.raw_pileup \
                .filter(pl.col("fraction_mod") > min_fraction_mod) \
                .filter(pl.col("modified").str.contains(modified)) \
                .filter(pl.col("score") > min_score)
        except Exception as e:
            raise ValueError(f"An error occurred while filtering the pileup: {str(e)}")

        self.pileup = pileup_filt


    def instantiate_sequence_enrichments(self, min_sites: int=4):
        """
        Function to instantiate motif identifiers using grouped pileup data.

        :param min_sites: Minimum number of sites
        :type min_sites: int, optional
        :returns: Dictionary of SequenceEnrichment
        :rtype: dict
        """

        # Instance validation
        assert isinstance(self.pileup, pl.DataFrame), "pileup must be a DataFrame"
        assert all(item in self.pileup.columns for item in ["strand", "ref", "modified", "sequence"]), \
            "Pileup DataFrame must contain 'strand', 'ref', 'modified' and 'sequence' columns"

        seqs_enrich = {}
        for group, data in self.pileup.groupby(["strand", "ref", "modified"]):
            if "sequence" in data.columns:
                if data.height < min_sites:
                    warnings.warn(f"Too few modification sites to process, {group}")
                    continue
                try:
                    sequences = data.get_column("sequence").to_list()
                    if group[0] == "-":
                        # Convert to reverse complement for negative strand
                        sequences = convert_flip_sequence(sequences)
                    # Create a SequenceEnrichment object for each group
                    seqs_enrich[group] = SequenceEnrichment(sequences)
                except ValueError as e:
                    warnings.warn(f"An error occurred while instantiating the SequenceEnrichment object of {group}: {str(e)}")
            else:
                warnings.warn(f"Data associated with {group} doesn't have a 'sequence' column.")

        return seqs_enrich
    
    def combine_sequence_enrichments_on_strand(self):
        """
        Function to combine motif identifiers on strand.

        :returns: Dictionary of SequenceEnrichment
        :rtype: dict

        :raises AssertionError: If SequenceEnrichment is not a dictionary

        """

        # Instance validation
        assert isinstance(self.sequence_enrichments_stranded, dict), "SequenceEnrichment must be a dictionary"

        merged = {}
        mods = self.pileup.get_column("modified").unique().to_list()
        refs = self.pileup.get_column("ref").unique().to_list()
        for mod in mods:
            for ref in refs:
                if ('-', ref, mod) in self.sequence_enrichments_stranded.keys():
                    rev_seq = self.sequence_enrichments_stranded['-', ref, mod].sequences
                else:
                    rev_seq = []
                if ('+', ref, mod) in self.sequence_enrichments_stranded.keys():
                    fwd_seq = self.sequence_enrichments_stranded['+', ref, mod].sequences
                else:
                    fwd_seq = []
                seq = [*rev_seq, *fwd_seq]
                if len(seq) > 1:
                    merged[(ref, mod)] = SequenceEnrichment([
                        *rev_seq,
                        *fwd_seq
                    ])
                else:
                    continue
        return merged
    
    def motif_modification_degree(self, motif: str, contig: str, mod_position: int, modification_type: str):
        # process forward strand
        mods_pos = np.array(
            self.pileup
                .filter(pl.col("ref") == contig)
                .filter(pl.col("strand") == "+")
                .filter(pl.col("modified") == modification_type)
                .get_column("start"))
        motif_distance_fwd = self.ref.smallest_distance_from_motif(motif, contig, mods_pos, reverse_strand=False)
        got_mod_fwd = motif_distance_fwd == mod_position

        # process reverse strand
        mods_pos = np.array(
            self.pileup
                .filter(pl.col("ref") == contig)
                .filter(pl.col("strand") == "-")
                .filter(pl.col("modified") == modification_type)
                .get_column("start"))
        motif_distance_rev = self.ref.smallest_distance_from_motif(motif, contig, mods_pos, reverse_strand=True)
        got_mod_rev = motif_distance_rev == (len(motif) - 1 - mod_position)
        return np.mean(np.concatenate([got_mod_fwd, got_mod_rev]))

    def sample_from_contig(self, contig, base_context = "ATGC", n=1000):
        return sample_seq_at_letter(self.ref.sequences[contig], n=n, size=self.window_size, letter=base_context)

    def sample_pssm(self, contig, base_context = "ATGC", **kwargs):
        """
        Sample position specific scoring matrix from a contig around a specific base.

        Parameters
        ----------
        contig : str
            Contig name
        base_context : str, optional
            Base to sample around, by default "ATGC" intpreted as A or T or G or C e.g. any base
        **kwargs : dict
            Keyword arguments to pass to sample_seq_at_letter

        Returns
        -------
        np.ndarray
            Position specific scoring matrix

        Raises
        ------
        ValueError
            If contig is not found in the reference file
        """
        samples_seqs = self.sample_from_contig(contig, base_context=base_context, **kwargs)
        pssm = SequenceEnrichment(samples_seqs).pssm()
        return pssm

    def update_kl_priors(self, prior_type="uniform", **kwargs):
        """
        Update the KL prior of the sequence_enrichments.

        :param prior_type: Type of prior, either "uniform", "gc" or "background"
        """
        assert prior_type in ["uniform", "gc", "background"], "prior_type must be either 'uniform', 'gc' or 'background'"
        mod_to_nuc = {
            "a": "A",
            "m": "C",
            "h": "C"
        }
        for i in self.sequence_enrichments:
            if prior_type == "uniform":
                prior = np.full_like(self.sequence_enrichments[i].pssm(), 0.25)
            elif prior_type == "background":
                prior = self.sample_pssm(i[0], base_context=mod_to_nuc[i[1]], n=len(self.sequence_enrichments[i].sequences))
            elif prior_type == "gc":
                prior = np.full_like(self.sequence_enrichments[i].pssm(), 0)
                gc = self.ref.seq_gc[i[0]]
                # Set AT frequency
                prior[0:2, :] = (1-gc)/2
                # Set GC frequency
                prior[2:4, :] = gc/2
            else:
                raise ValueError(f"Invalid prior_type: {prior_type}")
            self.sequence_enrichments[i].update_kl_prior(prior)
        self.kl_prior_type = prior_type

    def sample_kl_divergencies_from_ref_background(self, contig, mod_type, n_samples = 100, base_context = "ATGC",**kwargs):
        """
        Sample KL divergencies from methylated sequencies to a PSSM sampled from the contig with an equal number of sequences.

        Parameters
        ----------
        contig : str
            Contig name
        base_context : str, optional
            Base to sample around, by default "ATGC" intpreted as A or T or G or C e.g. any base
        **kwargs : dict
            Keyword arguments to pass to sample_seq_at_letter

        Returns
        -------
        np.ndarray
            KL divergencies
        """
        m = copy.copy(self.sequence_enrichments[(contig, mod_type)])
        sim = []
        for _ in range(n_samples):
            m.kl_prior = self.sample_pssm(contig, base_context, n=len(m.sequences))
            sim.append(m.kl_divergence())
        return sim
    

    def motif_candidates_methylation_frequency(self, contig, mod_type, min_kl=0.5):
        m = copy.copy(self.sequence_enrichments[(contig, mod_type)])
        candidates = m.get_motif_candidates(min_kl=min_kl)
        sequences = m.seq_to_int()[:, m.kl_divergence() > min_kl]

        motifs_freq = {}
        for i, motif in enumerate(candidates):
            fwd = self.ref.motif_positions(motif, contig).shape[0]
            rev = self.ref.motif_positions(motif, contig, reverse_strand=True).shape[0]
            n_ref = fwd + rev
            motif_int = [m.nuc_to_int[i] for i in [*motif] if i in m.nuc_to_int]
            n_mod = np.all(sequences == motif_int, axis=1).sum()
            motifs_freq[motif] =  {"n_mod": n_mod, "n_ref":n_ref, "freq":n_mod / n_ref}
        return motifs_freq
    
    def plot_kl_divergencies_from_ref_background(self, contig, mod_type, n_samples = 100, base_context = "ATGC", min_kl=0.2, **kwargs):
        """
        Plot KL divergencies from methylated sequencies to a PSSM sampled from the contig with an equal number of sequences.

        Parameters
        ----------
        contig : str
            Contig name
        base_context : str, optional
            Base to sample around, by default "ATGC" intpreted as A or T or G or C e.g. any base
        **kwargs : dict
            Keyword arguments to pass to sample_seq_at_letter

        Returns
        -------
        np.ndarray
            KL divergencies
        """
        # Sample KL divergencies
        sim = self.sample_kl_divergencies_from_ref_background(contig, mod_type, n_samples, base_context, **kwargs)
        m = copy.copy(self.sequence_enrichments[(contig, mod_type)])
        m.kl_prior = self.sample_pssm(contig, base_context, n=len(m.sequences))

        # Enrichment plot
        p, ax = m.plot_enrichment_with_prior(min_kl = min_kl)
        ax[0].set_title(ax[0].get_title() + f"\n{contig}");
        ax[1].set_title(ax[1].get_title() + "\nbackground");

        # Boxplot
        old_xlabs = ax[0].get_xticklabels()
        xtickpos = copy.copy([i.get_position()[0] for i in old_xlabs])
        xticklabs = copy.copy([i.get_text() for i in old_xlabs])
        bp = ax[0].boxplot(np.array(sim), positions=np.arange(0, m.seq_length), flierprops = dict(alpha = 0), patch_artist=True);
        ax[0].set_xticks(xtickpos);
        ax[0].set_xticklabels(xticklabs);
        for patch in bp['boxes']:
            patch.set_facecolor("lightgray");
            patch.set_edgecolor("black");
        for median in bp['medians']:
            median.set(color="black", linewidth=1);
    
    def plot_annotated_distance_heatmap(self, contig: str, mod_type: str, min_kl=0.2, min_group_size=10) -> None:
        m = self.sequence_enrichments[(contig, mod_type)]
        sequences = m.kl_masked_sequences(min_kl = min_kl)
        dist = edit_distance(m.sequences)
        heatmap = sns.clustermap(dist, cmap=sns.color_palette("YlOrBr", as_cmap=True));
        heatmap_data = heatmap.data2d.to_numpy()

        # Initialise dictionary of identical sequences
        identical_sequences_index = {} 
        group_start = 0
        identical_sequences_index[group_start] = [0]

        # Iterate over the diagonal of the heatmap
        for i in range(heatmap_data.shape[0] - 1):
            # if the next diagonal element is different, start a new group
            if heatmap_data[i, i] != heatmap_data[i+1, i]:
                group_start = i
                identical_sequences_index[group_start] = [i]
            else:
                identical_sequences_index[group_start] = identical_sequences_index[group_start] + [i]

        # Annotate heatmap with identical sequences
        for k, v in identical_sequences_index.items():
            if len(v) > min_group_size:
                seq = "".join((sequences[heatmap.dendrogram_col.reordered_ind[k]]))
                x_plot = v[int(len(v)/2)]
                contig_number_fwd = self.ref.motif_positions(seq, contig).shape[0]
                contig_number_rev = self.ref.motif_positions(seq, contig, reverse_strand=True).shape[0]
                contig_number = contig_number_fwd + contig_number_rev
                heatmap.figure.axes[2].text(x_plot, x_plot, f"{seq}\nmod={len(v)} | ref={contig_number} | {100*len(v)/contig_number:.2f}%", fontsize=8, ha='center', va="center", rotation=45)
        return heatmap
        
def convert_flip_sequence(seqs):
    """
    Function to convert the 5'->3' forward sequence to 5'->3' reverse sequence.

    :param seqs: List of sequences
    :type seqs: list
    :returns: Reverse complement of the sequences
    :rtype: list
    """

    # Input validations
    assert isinstance(seqs, list), "seqs must be a list"
    assert all(isinstance(item, str) for item in seqs), "All items in seqs must be strings"
    assert all(item.upper() in ["A", "C", "G", "T", "N"] for item in "".join(seqs)), \
        "All items in seqs must be valid DNA sequences"

    # Convert to complement
    seqs_rc = []
    for seq in seqs:
        seq_rc = ""
        for base in seq.upper()[::-1]:
            if base == "A":
                seq_rc += "T"
            elif base == "C":
                seq_rc += "G"
            elif base == "G":
                seq_rc += "C"
            elif base == "T":
                seq_rc += "A"
            elif base == "N":
                seq_rc += "N"
        seqs_rc.append(seq_rc)
    return seqs_rc


if __name__=="__main__":
    pass