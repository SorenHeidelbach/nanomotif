
import re
import numpy as np
import copy
from nanomotif.utils import distance_to_nearest_value_in_arrays
from nanomotif.dna import DNAsequences
import nanomotif.dna as dna

class Reference:
    def __init__(self, fasta_path: str = None, sequence_dictionary: dict = None, **kwargs):
        if sequence_dictionary is not None:
            self.sequences = self._str_dict_to_dna_dict(sequence_dictionary)
        elif fasta_path is not None:
            self.sequences = self.read_from_fasta(fasta_path, **kwargs)
        else:
            raise ValueError("Either fasta_path or sequence_dictionary must be provided")
        
        self.seq_lengths = {key: len(val[0]) for key, val in self.sequences.items()}
        self.seq_gc = {key: val.gc()[0] for key, val in self.sequences.items()}
    
    @property
    def sequences(self):
        return self._sequences
    
    @sequences.setter
    def sequences(self, sequences):
        self._check_sequences(sequences)
        self._sequences = sequences

    def _check_sequences(self, sequences):
        assert type(sequences) == dict
        assert len(sequences) > 0
        for k, v in sequences.items():
            assert type(k) == str
            assert type(v) == DNAsequences

    def _str_dict_to_dna_dict(self, sequences):
        # Updates the sequences dictionary to be a dictionary of DNAsequences in place
        return {k: DNAsequences([v]) for k, v in sequences.items()}

    def read_from_fasta(self, path, trim_names=True, trim_character=" ") -> dict:
        """
        Reads a fasta file and returns a dictionary with the contig names as 
        keys and the sequences as values
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        data = {}
        active_sequence_name = "no_header"
        for line in lines:
            line = line.rstrip()
            if line[0] == '>':
                active_sequence_name = line[1:]
                if trim_names:
                    active_sequence_name = active_sequence_name.split(trim_character)[0]
                if active_sequence_name not in data:
                    data[active_sequence_name] = ''
            else:
                data[active_sequence_name] += line
        return self._str_dict_to_dna_dict(data)
    
    def get_seqeunce_names(self) -> list:
        return list(self.sequences.keys())
    
    def get_surrounding_sequence(self, contig: str, position: int, window_size: int = 10, padding: str="N") -> str:
        '''Returns the sequence of a contig from a fasta dictionary'''
        assert contig in self.sequences.keys(), "Contig not found in fasta file"
        if position < 0 or position > self.seq_lengths[contig]:
            raise ValueError(f"Position {position} is out of bounds for contig {contig} of length {self.seq_lengths[contig]}")
        if position - window_size < 0 or position + window_size > self.seq_lengths[contig]:
            start = position - window_size
            start_clipped = max(start, 0)
            end = position + window_size
            end_clipped = min(end, self.seq_lengths[contig])
            seq = padding * (start_clipped - start) + self.sub_seq(contig, start_clipped, end_clipped) + padding * (end - end_clipped + 1)
        else:
            seq = self.sub_seq(contig, position - window_size, position + window_size + 1)
        return seq

    def sub_seq(self, contig: str, start: int = None, end: int = None) -> str:
        '''Returns the sequence of a contig from a fasta dictionary'''
        return self.sequences[contig][0][start:end]
    
    def motif_positions(self, motif: str, contig: str, reverse_strand=False) -> np.ndarray:
        '''Returns a list of the positions of a motif in a sequence in 1-based indexing'''

        if reverse_strand:
            seq = self.sequences[contig].reverse_complement()[0]
        else:
            seq = self.sequences[contig][0]
        positions = []
        for match in re.finditer(motif, seq):
            positions.append(match.start())
        positions = np.array(positions, dtype=np.int64)

        if reverse_strand:
            positions = self.seq_lengths[contig] - positions - len(motif)

        return positions

    def smallest_distance_to_motif(self, motif: str, contig: str, positions: np.ndarray, reverse_strand=False) -> np.ndarray:
        '''Returns the smallest distance between from positions and the start of the closest motif'''
        motif_start = np.array(self.motif_positions(motif, contig, reverse_strand=reverse_strand))
        if motif_start.shape[0] == 0:
            return np.full_like(positions, np.nan)
        else:
            return np.array(distance_to_nearest_value_in_arrays(positions, motif_start))
        
    def smallest_distance_from_motif(self, motif: str, contig: str, positions: np.ndarray, reverse_strand=False) -> np.ndarray:
        '''Returns the smallest distance between from the start of the motif to positions '''
        motif_start = self.motif_positions(motif, contig, reverse_strand=reverse_strand)
        if motif_start.shape[0] == 0:
            return np.array([])
        else:
            return np.array(distance_to_nearest_value_in_arrays(motif_start, positions))

    def write_fasta(self, out: str):
        with open(out, 'w') as f:
            for name, seq in self.sequences.items():
                f.write(f">{name}\n")
                f.write(f"{seq[0]}\n")



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)