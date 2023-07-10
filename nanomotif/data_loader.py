import os
import polars as pl

class DataLoader():
    """
    A DataLoader class that loads data from pileup and reference files, and handles output file operations.
    """
    def __init__(self, pileup_file):
        assert isinstance(pileup_file, str), "pileup_file must be a string"
        if not os.path.isfile(pileup_file):
            raise FileNotFoundError(f"pileup_file {pileup_file} does not exist")
        
        self.pileup_file = pileup_file

        self.pileup = self.load_modkit_pileup_bed()
    

    def load_modkit_pileup_bed(self):
        """
        Load bed file from modkit pileup. 
        """
        header_names = (
            "ref",
            "start",
            "modified",
            "score",
            "strand",
            "fraction_mod"
        )

        try:
            number_of_columns = pl.read_csv(self.pileup_file, separator="\t", has_header=False, n_rows=1).width

            if number_of_columns == 10:
                print("The pileup file has been outputted with tabs seperators in the last fields.")
                pileup = pl.read_csv(self.pileup_file, separator="\t", has_header=False, columns=[0,1,3,4,5,9])
                pileup = pileup.with_columns(
                    pl.col("column_10")
                        .str.split(by=" ").list.get(1)
                        .cast(pl.Float64, strict=False).alias("fraction_mod")
                ).drop("column_10")
                pileup.columns = header_names
            else:
                pileup = pl.read_csv(self.pileup_file, separator="\t", has_header=False, columns=[0,1,3,4,5,10])
                pileup.columns = header_names
        except Exception as e:
            raise ValueError(f"An error occurred while reading the pileup file: {str(e)}")
        return pileup



