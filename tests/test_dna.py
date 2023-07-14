

import pytest
from nanomotif.dna import *
from hypothesis import given, strategies as st
import random
import string
import itertools
import pandas as pd
import polars.testing

class TestDNAsequences:
    # Test that the function raises an AssertionError when sequences is not a list
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=0, max_size=100))
    def test_sequences_type(self, sequences):
        with pytest.raises(AssertionError, match="DNA sequences must be a list"):
            DNAsequences(sequences)

    # Test that the function raises an AssertionError when sequences is an empty list
    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=100), min_size=0, max_size=0))
    def test_sequences_empty(self, sequences):
        with pytest.raises(AssertionError, match="DNA sequences must not be empty"):
            DNAsequences(sequences)

    # Test that the function raises an AssertionError when sequences contains non-string elements
    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=100))
    def test_sequences_elements(self, sequences):
        with pytest.raises(AssertionError, match="DNA sequences must be a list of strings"):
            DNAsequences(sequences)

    # Test that the function raises an AssertionError when sequences contains empty strings
    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=0, max_size=0), min_size=1, max_size=100))
    def test_sequences_elements_empty(self, sequences):
        with pytest.raises(AssertionError, match="DNA sequences must not be empty strings"):
            DNAsequences(sequences)

    @given(st.lists(st.text(alphabet=DNAsequences.bases, min_size=1), min_size=1))
    def test_reverse_complement(self, dna_sequences):
        seq = DNAsequences(dna_sequences)
        reversed_complemented = seq.reverse_complement()

        # check that the reverse complement sequences have the same length as the original sequences
        assert len(reversed_complemented) == len(dna_sequences)
        
        # check that each reverse complement sequence is indeed the reverse complement of the original sequence
        for original, rev_comp in zip(dna_sequences, reversed_complemented):
            rev_comp_reversed = "".join(reversed(rev_comp))
            assert all(seq.complement[base] == rev_base for base, rev_base in zip(original, rev_comp_reversed))

    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=10), min_size=1, max_size=200))
    def test_gc_content(self, sequences):
        # Create DNAsequences objects from your test data
        seqs = DNAsequences(sequences)

        # Get GC content using the method
        gc_content = seqs.gc()

        # Compute the GC content manually for comparison
        expected_gc_content = [(seq.count("G") + seq.count("C")) / len(seq) for seq in sequences]

        # Use numpy's isclose function to compare two arrays of floating point numbers with a tolerance
        assert np.all(np.isclose(gc_content, expected_gc_content, atol=1e-8))

    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1), min_size=1, max_size=200))
    def test_trimmed_sequences_length(self, sequences):
        # Create DNAsequences object from your test data
        seqs = DNAsequences(sequences)

        # Get trimmed sequences using the method
        trimmed_seqs = seqs.trimmed_sequences()

        # Compute the minimum length of the sequences
        min_length = min(len(seq) for seq in sequences)

        # Check that all trimmed sequences have the correct length
        for seq in trimmed_seqs:
            assert len(seq) == min_length, f"Trimmed sequence length {len(seq)} is not equal to minimum length {min_length}"
        
    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1), min_size=1, max_size=200))
    def test_trimmed_sequences_content(self, sequences):
        # Create DNAsequences object from your test data
        seqs = DNAsequences(sequences)

        # Get trimmed sequences using the method
        trimmed_seqs = seqs.trimmed_sequences()

        # Compute the minimum length of the sequences
        min_length = min(len(seq) for seq in sequences)

        # Check that all trimmed sequences have the correct content
        for original, trimmed in zip(sequences, trimmed_seqs):
            assert original[:min_length] == trimmed, f"Trimmed sequence {trimmed} is not equal to the start of original sequence {original[:min_length]}"
    
    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=200), min_size=1, max_size=200), st.text(alphabet=["N"], min_size=1, max_size=1))
    def test_padded_sequences_length(self, sequences, pad):
        # Create DNAsequences object from your test data
        seqs = DNAsequences(sequences)

        # Get padded sequences using the method
        padded_seqs = seqs.padded_sequences(pad)

        # Compute the maximum length of the sequences
        max_length = max(len(seq) for seq in sequences)

        # Check that all padded sequences have the correct length
        for seq in padded_seqs:
            assert len(seq) == max_length, f"Padded sequence length {len(seq)} is not equal to maximum length {max_length}"

    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=200), min_size=1, max_size=200), st.text(alphabet=["N", "A", "T", "C", "G", "H", "V"], min_size=1, max_size=1))
    def test_padded_sequences_content(self, sequences, pad):
        # Create DNAsequences object from your test data
        seqs = DNAsequences(sequences)

        # Get padded sequences using the method
        padded_seqs = seqs.padded_sequences(pad)

        # Compute the maximum length of the sequences
        max_length = max(len(seq) for seq in sequences)

        # Check that all padded sequences have the correct content
        for original, padded in zip(sequences, padded_seqs):
            assert original + pad * (max_length - len(original)) == padded, f"Padded sequence {padded} is not equal to the original sequence {original} with padding {pad}"
        

class TestGenerateRandomDnaSequence:
    # Test that the function returns the correct length
    def test_sequence_length(self):
        seq_len = 10
        seq = generate_random_dna_sequence(seq_len)
        assert len(seq) == seq_len, "The length of the sequence is not correct"

    # Test that the sequence only contains characters from the provided alphabet
    def test_sequence_alphabet(self):
        alphabet = ['A', 'C', 'G', 'T']
        seq = generate_random_dna_sequence(100, alphabet)
        assert all(char in alphabet for char in seq), "The sequence contains characters not in the alphabet"

    # Test that the function works with a different alphabet
    def test_different_alphabet(self):
        alphabet = ['0', '1']
        seq = generate_random_dna_sequence(100, alphabet)
        assert all(char in alphabet for char in seq), "The sequence contains characters not in the alternate alphabet"

    # Test that the function returns an empty string when length is 0
    def test_zero_length(self):
        seq = generate_random_dna_sequence(0)
        assert seq == "", "The sequence is not empty when the length is 0"

    # Property-based testing with hypothesis
    @given(st.integers(min_value=1, max_value=1000), st.lists(st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=26))
    def test_hypothesis(self, length, alphabet):
        seq = generate_random_dna_sequence(length, alphabet)
        assert len(seq) == length, "The length of the sequence is not correct"
        assert all(char in alphabet for char in seq), "The sequence contains characters not in the alphabet"



class TestPadDnaSeq:
    # Test that the function returns the correct length
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=100), st.integers(min_value=1, max_value=200))
    def test_pad_length(self, seq, target_length):
        if len(seq) > target_length:
            with pytest.raises(AssertionError):
                pad_dna_seq(seq, target_length)
        else:
            padded_seq = pad_dna_seq(seq, target_length)
            assert len(padded_seq) == target_length

    # Test that the padded sequence starts with the original sequence when padded from the left
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=100), st.integers(min_value=1, max_value=200))
    def test_pad_left(self, seq, target_length):
        if len(seq) <= target_length:
            padded_seq = pad_dna_seq(seq, target_length, where="left")
            assert padded_seq.startswith(seq)

    # Test that the padded sequence ends with the original sequence when padded from the right
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=100), st.integers(min_value=1, max_value=200))
    def test_pad_right(self, seq, target_length):
        if len(seq) <= target_length:
            padded_seq = pad_dna_seq(seq, target_length, where="right")
            assert padded_seq.endswith(seq)

    # Test that the original sequence is in the middle when padded from both sides
    # Side effect test, whether padding is added left first, then right in case of odd number of padding
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=100), st.integers(min_value=1, max_value=200))
    def test_pad_sides(self, seq, target_length):
        if len(seq) <= target_length:
            padded_seq = pad_dna_seq(seq, target_length, where="sides")
            pad_length_right = (target_length - len(seq)) // 2
            pad_length_left = target_length - len(seq) - pad_length_right
            assert padded_seq[pad_length_left:-pad_length_right or None] == seq



class TestPadDnaSeqN:
    # Test that the function returns the correct length
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=10), st.integers(min_value=1, max_value=20))
    def test_pad_length(self, seq, target_length):
        if len(seq) > target_length:
            with pytest.raises(AssertionError):
                pad_dna_seq_n(seq, target_length)
        else:
            padded_seq = pad_dna_seq_n(seq, target_length)
            assert len(padded_seq) == target_length

    # Test that the sequence only contains characters from the original sequence and 'N'
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=10), st.integers(min_value=1, max_value=20))
    def test_sequence_contents(self, seq, target_length):
        if len(seq) <= target_length:
            padded_seq = pad_dna_seq_n(seq, target_length)
            assert all(char in (list(seq) + ['N']) for char in padded_seq)

    # Test that the original sequence is in the middle when padded from both sides
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=10), st.integers(min_value=1, max_value=20))
    def test_pad_sides(self, seq, target_length):
        if len(seq) <= target_length:
            padded_seq = pad_dna_seq(seq, target_length, where="sides")
            pad_length_right = (target_length - len(seq)) // 2
            pad_length_left = target_length - len(seq) - pad_length_right
            assert padded_seq[pad_length_left:-pad_length_right or None] == seq

    # Test that the padded sequence starts with the original sequence when padded from the left
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=10), st.integers(min_value=1, max_value=20))
    def test_pad_left(self, seq, target_length):
        if len(seq) <= target_length:
            padded_seq = pad_dna_seq_n(seq, target_length, where="left")
            assert padded_seq.startswith(seq)

    # Test that the padded sequence ends with the original sequence when padded from the right
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=10), st.integers(min_value=1, max_value=20))
    def test_pad_right(self, seq, target_length):
        if len(seq) <= target_length:
            padded_seq = pad_dna_seq_n(seq, target_length, where="right")
            assert padded_seq.endswith(seq)



class TestGenerateAllPossibleKmers:
    # Test the count of generated k-mers is correct
    @given(st.integers(min_value=1, max_value=5), st.lists(st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=4))
    def test_kmer_count(self, k, alphabet):
        kmers = generate_all_possible_kmers(k, alphabet)
        assert len(kmers) == len(alphabet)**k

    # Test all k-mers are of the correct length
    @given(st.integers(min_value=1, max_value=5), st.lists(st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=4))
    def test_kmer_length(self, k, alphabet):
        kmers = generate_all_possible_kmers(k, alphabet)
        assert all(len(kmer) == k for kmer in kmers)

    # Test all possible k-mers are generated
    @given(st.integers(min_value=1, max_value=5), st.lists(st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=4))
    def test_all_kmers(self, k, alphabet):
        kmers = generate_all_possible_kmers(k, alphabet)
        all_possible_kmers = ["".join(kmer) for kmer in itertools.product(alphabet, repeat=k)]
        assert set(kmers) == set(all_possible_kmers)



class TestConvertIntsToBases:
    conversion_dict = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    # Test that the function converts a list of integers to a string of bases correctly
    @given(st.lists(st.integers(min_value=0, max_value=3), max_size=100))
    def test_convert_ints(self, ints):
        expected_result = ''.join(self.conversion_dict[i] for i in ints)
        result = convert_ints_to_bases(ints, self.conversion_dict)
        assert result == expected_result

    # Test that the function raises a KeyError when an integer is not in the conversion_dict
    @given(st.lists(st.integers(min_value=0, max_value=4), max_size=100))
    def test_key_error(self, ints):
        if any(i not in self.conversion_dict for i in ints):
            with pytest.raises(KeyError):
                convert_ints_to_bases(ints, self.conversion_dict)
                
    # Test the function works with an empty list
    @given(st.lists(st.integers(min_value=0, max_value=3), max_size=0))
    def test_empty_list(self, ints):
        expected_result = ''
        result = convert_ints_to_bases(ints, self.conversion_dict)
        assert result == expected_result



class TestSampleSeqAtLetter:
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=100), st.integers(min_value=1), st.integers(min_value=1), st.sampled_from(['A', 'T', 'C', 'G']))
    def test_sample_size(self, seq, n, size, letter):
        result = sample_seq_at_letter(seq, n, size, letter)
        assert all(len(subseq) == 2 * size + 1 for subseq in result)

    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1, max_size=1000), st.integers(min_value=1), st.integers(min_value=1), st.sampled_from(['A', 'T', 'C', 'G']))
    def test_correct_sampling(self, seq, n, size, letter):
        result = sample_seq_at_letter(seq, n, size, letter)
        for subseq in result:
            middle = len(subseq) // 2
            assert subseq[middle] == letter

    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1), st.integers(min_value=1), st.integers(min_value=1), st.sampled_from(['A', 'T', 'C', 'G']))
    def test_empty_positions(self, seq, n, size, letter):
        positions = [i for i, ltr in enumerate(seq) if ltr in letter and i-size >= 0 and i+size+1 < len(seq)]
        if len(positions) == 0:
            result = sample_seq_at_letter(seq, n, size, letter)
            assert result == []



class TestGetKmersInSequence:
    # Test that function generates correct k-mers
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1), st.integers(min_value=1, max_value=10))
    def test_kmers_generation(self, sequence, kmer_size):
        result = get_kmers_in_sequence(sequence, kmer_size)
        if kmer_size > len(sequence):
            assert result == []
        else:
            assert result == [sequence[i:i+kmer_size] for i in range(len(sequence) - kmer_size + 1)]

    # Test that function raises an AssertionError when sequence is not a string
    @given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10))
    def test_sequence_type(self, sequence, kmer_size):
        with pytest.raises(AssertionError, match="sequence must be a string"):
            get_kmers_in_sequence(sequence, kmer_size)

    # Test that function raises an AssertionError when kmer_size is not an integer
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1), st.floats(min_value=1, max_value=10))
    def test_kmer_size_type(self, sequence, kmer_size):
        with pytest.raises(AssertionError, match="kmer_size must be an integer"):
            get_kmers_in_sequence(sequence, kmer_size)

    # Test that function raises an AssertionError when kmer_size is less than 1
    @given(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=1), st.integers(min_value=-10, max_value=0))
    def test_kmer_size_value(self, sequence, kmer_size):
        with pytest.raises(AssertionError, match="kmer_size must be greater than 0"):
            get_kmers_in_sequence(sequence, kmer_size)



class TestKmerGraph:
    # Test that function raises an AssertionError when sequences are shorter than kmer_size + stride
    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], max_size=3), min_size=1), st.integers(min_value=3, max_value=10), st.integers(min_value=2, max_value=10), st.booleans())
    def test_short_sequences(self, sequences, kmer_size, stride, position_aware):
        with pytest.raises(AssertionError):
            kmer_graph(sequences, kmer_size, stride, position_aware)

    # Test that function produces correct kmer graph
    @given(st.lists(st.text(alphabet=['A', 'T', 'C', 'G'], min_size=10), min_size=1, max_size=200), st.integers(min_value=2, max_value=5), st.integers(min_value=1, max_value=5), st.booleans())
    def test_correct_kmer_graph(self, sequences, kmer_size, stride, position_aware):
        result = kmer_graph(sequences, kmer_size, stride, position_aware)

        # Define expected kmer graph
        seqs = DNAsequences(sequences)
        all_kmers = seqs.kmers(kmer_size)
        kmers_from = [k[::stride][:-1] for k in all_kmers]
        kmers_to = [k[::stride][1:] for k in all_kmers]
        positions = [i*stride for kmer in kmers_from for i in range(len(kmer))]
        expected_kmer_graph = pl.DataFrame({
            "from": utils.flatten_list(kmers_from), 
            "to": utils.flatten_list(kmers_to), 
            "position": positions
        })
        if position_aware:
            expected_kmer_graph = expected_kmer_graph.groupby(["from", "to", "position"]).agg(pl.count().alias("count")).sort("from", "to", "count", "position")
        else:
            expected_kmer_graph = expected_kmer_graph.groupby(["from", "to"]).agg(pl.count().alias("count")).sort("from", "to", "count")
        
        expected_kmer_graph = expected_kmer_graph

        polars.testing.assert_frame_equal(result, expected_kmer_graph)

if __name__ == '__main__':
    pytest.main()
