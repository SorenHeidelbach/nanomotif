import pytest
import nanomotif as nm

def test_load_modkit_pileup_bed():
    nm.data_loader.DataLoader("tests/data/sample_pileup.bed").load_modkit_pileup_bed().columns == ['ref', 'start', 'modified', 'score', 'strand', 'fraction_mod']
