import pytest
import nanomotif as mb
import polars as pl

def test_load_modkit_pileup_bed():
    mb.data_loader.DataLoader("/home/ubuntu/vol_store/nanomotif/scripts/tests/data/sample_pileup.bed").load_modkit_pileup_bed().columns == ['ref', 'start', 'modified', 'score', 'strand', 'fraction_mod']
