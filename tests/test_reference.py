import pytest
import numpy as np
from hypothesis import given, strategies as st
from nanomotif.reference import Reference  # Assuming your class is in reference.py
from nanomotif.dna import DNAsequences
# Hypothesis strategy for generating valid DNA sequences
dna_seq_strategy = st.text(alphabet="ACGT", min_size=1, max_size=10000)

class TestReference:

    fasta_path = 'tests/data/test.fasta'
    
    @given(dna_seq=dna_seq_strategy)
    def test_init_with_sequence_dictionary(self, dna_seq):
        seq_dict = {"seq1": dna_seq}
        ref = Reference(sequence_dictionary=seq_dict)
        assert isinstance(ref.sequences, dict)
        for k, v in ref.sequences.items():
            assert isinstance(k, str)
            assert isinstance(v, DNAsequences)

    def test_init_with_fasta_path(self):
        ref = Reference(fasta_path=self.fasta_path)
        assert isinstance(ref.sequences, dict)
        for k, v in ref.sequences.items():
            assert isinstance(k, str)
            assert isinstance(v, DNAsequences)

    def test_init_no_args(self):
        with pytest.raises(ValueError):
            ref = Reference()

    def test_get_sequence_names(self):
        ref = Reference(fasta_path=self.fasta_path)
        assert len(ref.get_seqeunce_names()) == len(ref.sequences)

    @given(start=st.integers(min_value=0), end=st.integers(min_value=1))
    def test_sub_seq(self, start, end):
        if start < end:
            ref = Reference(fasta_path=self.fasta_path)
            # Choose a sequence (in this case, the first one) to test the sub_seq method
            contig = ref.get_seqeunce_names()[0]
            sequence = ref.sequences[contig][0]
            assert ref.sub_seq(contig, start, end) == sequence[start:end]

    @given(st.fixed_dictionaries({
        "trim_names": st.booleans(),
        "trim_character": st.text(alphabet="_ ", min_size=1, max_size=1)
    }))
    def test_header_trimming(self, kwargs):
        ref = Reference(fasta_path=self.fasta_path, **kwargs)
        if kwargs["trim_names"]:
            if kwargs["trim_character"] == "_":
                assert ref.get_seqeunce_names()[2] == "Sequence3"
            elif kwargs["trim_character"] == " ":
                assert ref.get_seqeunce_names()[3] == "Sequence4"
        else:
            assert ref.get_seqeunce_names()[2] == "Sequence3_underline_trim"
            assert ref.get_seqeunce_names()[3] == "Sequence4 space trim"

    def test_get_seq_lengths(self):
        ref = Reference(fasta_path=self.fasta_path)
        assert list(ref.seq_lengths.values()) == [1000, 210, 800, 900]

    def test_get_seqeunce_names(self):
        ref = Reference(fasta_path=self.fasta_path)
        assert ref.get_seqeunce_names() == ["Sequence1_no_linebreak", "Sequence2_linebreak", "Sequence3_underline_trim", "Sequence4"]

    @given(st.text(), st.integers(0, 100), st.integers(1, 100), st.text(alphabet="ACGT", min_size=1, max_size=1))
    def test_get_surrounding_sequence(self, contig, position, window_size, padding):
        # Generate test input using the strategies defined
        ref = Reference(sequence_dictionary={contig: 'ACTGGATC'*100})

        # Call the method under test
        result = ref.get_surrounding_sequence(contig, position, window_size, padding)

        # Perform assertions on the result or any other desired checks
        assert len(result) == (2 * window_size) + 1
    
    @given(st.text(alphabet="ACGT", min_size=1, max_size=4))
    def test_motif_positions(self, motif):
        motif_padded = 'NNNN' + motif + 'NNNN'
        ref = Reference(sequence_dictionary={"TEST": motif_padded*100})
        result = ref.motif_positions(motif, "TEST")
        expected = np.arange(4, len(motif_padded)*100, len(motif_padded))
        assert np.all(result == expected)
    
    @given(st.text(alphabet="ACGT", min_size=1, max_size=4))
    def test_motif_positions_reverse(self, motif):
        reverse_motif = DNAsequences([motif]).reverse_complement()[0]
        motif_padded = 'NNNN' + motif + 'NNNN'
        ref = Reference(sequence_dictionary={"TEST": motif_padded*100})
        result = ref.motif_positions(reverse_motif, "TEST", reverse_strand=True)
        expected = np.arange(4, len(motif_padded)*100, len(motif_padded))[::-1]
        assert np.all(result == expected)
    