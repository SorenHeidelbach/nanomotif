# nanomotif

## Intro

The project aims to develop a method for binning microbial contigs from an assembly into bins based on the methylation status. 

Methylation status is used to denote which sequence motif are possibly modified with a nucleotide modification (6mA or 5mC) and the information used for binning.

The repo contains several part. 

- Identification of possible modified motif candidates
- Representation of motif status for binning


## Overview

The analysis is generally divided into three parts:

1. Generation of positional modification status on refrence/assembly sequences
2. Extraction of methylated motif in reference sequences
3. Clustering of sequences based on methylation motifs

### Step 1
The output of `dorado` is mapped to the reference/assembly. The mappings are piled up using `modkit`, generating positional modification status for each modification type for each reference/assembly sequence.

### Step 2
Based on the sequence context of modified positions on the contig, modified motifs are extracted.

### Step 3
The degree of motif modification is used as a feature for clustering contigs in the assembly. 


