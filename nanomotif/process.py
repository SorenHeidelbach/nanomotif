from nanomotif.data_loader import DataLoader
from nanomotif.utils import Sequences
from nanomotif.SequenceEnrichment import SequenceEnrichment
from nanomotif import utils
import numpy as np
import warnings
import polars as pl
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

        self.ref = Sequences(ref_file)
        self.raw_pileup = DataLoader(pileup_file).pileup 
        self.filter_pileup(min_fraction_mod, modified, min_score)
        self.pileup = self.add_ref_sequence_at_modification(window_size) 
        self.SequenceEnrichment_stranded = self.instantiate_motif_identifiers()
        self.SequenceEnrichment = self.combine_SequenceEnrichment_identifiers_on_strand()
        self.update_SequenceEnrichment_kl_prior()
        self.window_size = window_size

        # Instance validation
        assert isinstance(self.pileup, pl.DataFrame), "pileup must be a DataFrame"
        assert isinstance(self.ref, Sequences), "ref must be a Sequences object"

    def update_motif_identifiers(self):
        """
        Function to update motif identiers, reflecting the current pileup date.
        """

        self.SequenceEnrichment_stranded = self.instantiate_motif_identifiers()
        self.SequenceEnrichment = self.combine_SequenceEnrichment_identifiers_on_strand()

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
        assert isinstance(self.ref, Sequences), "ref must be a Sequences object"

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


    def instantiate_motif_identifiers(self, min_sites: int=4):
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

        SequenceEnrichment = {}
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
                    SequenceEnrichment[group] = SequenceEnrichment(sequences)
                except ValueError as e:
                    warnings.warn(f"An error occurred while instantiating the SequenceEnrichment object of {group}: {str(e)}")
            else:
                warnings.warn(f"Data associated with {group} doesn't have a 'sequence' column.")

        return SequenceEnrichment
    
    def combine_SequenceEnrichment_identifiers_on_strand(self):
        """
        Function to combine motif identifiers on strand.

        :returns: Dictionary of SequenceEnrichment
        :rtype: dict

        :raises AssertionError: If SequenceEnrichment is not a dictionary

        """

        # Instance validation
        assert isinstance(self.SequenceEnrichment_stranded, dict), "SequenceEnrichment must be a dictionary"

        merged = {}
        mods = self.pileup.get_column("modified").unique().to_list()
        refs = self.pileup.get_column("ref").unique().to_list()
        for mod in mods:
            for ref in refs:
                if ('-', ref, mod) in self.SequenceEnrichment_stranded.keys():
                    rev_seq = self.SequenceEnrichment_stranded['-', ref, mod].sequences
                else:
                    rev_seq = []
                if ('+', ref, mod) in self.SequenceEnrichment_stranded.keys():
                    fwd_seq = self.SequenceEnrichment_stranded['+', ref, mod].sequences
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
        return utils.sample_seq_at_letter(self.ref.sequences[contig], n=n, size=self.window_size, letter=base_context)

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

    def update_SequenceEnrichment_kl_prior(self, prior_type="uniform", **kwargs):
        """
        Update the KL prior of the SequenceEnrichment.

        :param prior_type: Type of prior, either "uniform", "gc" or "background"
        """
        assert prior_type in ["uniform", "gc", "background"], "prior_type must be either 'uniform', 'gc' or 'background'"
        mod_to_nuc = {
            "a": "A",
            "m": "C",
            "h": "C"
        }
        for i in self.SequenceEnrichment:
            if prior_type == "uniform":
                prior = np.full_like(self.SequenceEnrichment[i].pssm(), 0.25)
            elif prior_type == "background":
                prior = self.sample_pssm(i[0], base_context=mod_to_nuc[i[1]], n=len(self.SequenceEnrichment[i].sequences))
            elif prior_type == "gc":
                prior = np.full_like(self.SequenceEnrichment[i].pssm(), 0)
                gc = self.ref.seq_gc[i[0]]
                # Set AT frequency
                prior[0:2, :] = (1-gc)/2
                # Set GC frequency
                prior[2:4, :] = gc/2
            else:
                raise ValueError(f"Invalid prior_type: {prior_type}")
            self.SequenceEnrichment[i].update_kl_prior(prior)

        
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