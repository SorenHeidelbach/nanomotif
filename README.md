# Nanomotif

Nanomotif is a Python package that provides functionality for analyzing and manipulating DNA sequences. It is designed to help researchers in the field of nanotechnology and DNA nanotechnology to analyze and work with DNA sequences more effectively. With Nanomotif, you can perform tasks such as motif identification, sequence enrichment analysis, and sequence manipulation.

## Installation

### Local Environment

To install Nanomotif in your local Python environment, follow these steps:

```shell
git clone https://github.com/your-username/nanomotif.git
cd nanomotif
pip install -r requirements.txt
pip install .
```

### Conda Environment

If you prefer using Conda for managing your Python environments, you can create a new environment and install Nanomotif as follows:

```shell
conda create -n nanomotif-env python=3.9
conda activate nanomotif-env
git clone https://github.com/your-username/nanomotif.git
cd nanomotif
conda install --file requirements.txt
python -m pip install .
```

## Example Usage

Here's an example to demonstrate how to use Nanomotif to perform sequence enrichment analysis:


```python
from nanomotif.motifs import SequenceEnrichment
from nanomotif import dna
import matplotlib.pyplot as plt
sequences = [ 
        dna.generate_random_dna_sequence(3) +
        dna.generate_random_dna_sequence(1, alphabet=["G"]*97 + ["T", "A", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["A"]*97 + ["T", "C", "G"]) +
        dna.generate_random_dna_sequence(1, alphabet=["T"]*97 + ["G", "C", "T"]) +
        dna.generate_random_dna_sequence(1, alphabet=["C"]*97 + ["C", "A", "G"]) +
        dna.generate_random_dna_sequence(3)
        for _ in range(200)
    ]
seq_enrichment = SequenceEnrichment(sequences)

# Calculate the positional frequency of each nucleotide
seq_enrichment.pssm()
```




    array([[0.3175, 0.2675, 0.2325, 0.0225, 0.9725, 0.0025, 0.0225, 0.2475,
            0.3225, 0.2375],
           [0.2125, 0.2125, 0.2375, 0.0075, 0.0075, 0.9775, 0.0025, 0.2625,
            0.2225, 0.2875],
           [0.2225, 0.2525, 0.2825, 0.9775, 0.0125, 0.0175, 0.0175, 0.2175,
            0.2225, 0.2775],
           [0.2575, 0.2775, 0.2575, 0.0025, 0.0175, 0.0125, 0.9675, 0.2825,
            0.2425, 0.2075]])




```python
# Get the consensus sequence
seq_enrichment.consensus()
```




    'TTTGATCAAT'




```python
dna.plot_dna_sequences(sequences[0:10])
```




    <Axes: >




    
![png](README_files/README_3_1.png)
    



```python
# Plot positional enrichment of each position
seq_enrichment.plot_enrichment()
plt.legend();
```


    
![png](README_files/README_4_0.png)
    


Extract K-mer connection in the sequence


```python
exmaple_kmer_graph = dna.kmer_graph(sequences, kmer_size=2)
exmaple_kmer_graph
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (207, 4)</small><table border="1" class="dataframe"><thead><tr><th>from</th><th>to</th><th>position</th><th>count</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>u32</td></tr></thead><tbody><tr><td>&quot;AA&quot;</td><td>&quot;AA&quot;</td><td>6</td><td>1</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AA&quot;</td><td>2</td><td>2</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AA&quot;</td><td>0</td><td>3</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AA&quot;</td><td>7</td><td>4</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AC&quot;</td><td>6</td><td>1</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AC&quot;</td><td>7</td><td>2</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AC&quot;</td><td>0</td><td>8</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AG&quot;</td><td>0</td><td>5</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AG&quot;</td><td>7</td><td>7</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AG&quot;</td><td>1</td><td>9</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AT&quot;</td><td>1</td><td>1</td></tr><tr><td>&quot;AA&quot;</td><td>&quot;AT&quot;</td><td>6</td><td>1</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>&quot;TG&quot;</td><td>&quot;GT&quot;</td><td>0</td><td>2</td></tr><tr><td>&quot;TG&quot;</td><td>&quot;GT&quot;</td><td>7</td><td>4</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TA&quot;</td><td>7</td><td>1</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TA&quot;</td><td>0</td><td>2</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TC&quot;</td><td>4</td><td>1</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TC&quot;</td><td>0</td><td>3</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TC&quot;</td><td>7</td><td>3</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TG&quot;</td><td>7</td><td>3</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TG&quot;</td><td>0</td><td>4</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TG&quot;</td><td>1</td><td>5</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TT&quot;</td><td>0</td><td>1</td></tr><tr><td>&quot;TT&quot;</td><td>&quot;TT&quot;</td><td>7</td><td>5</td></tr></tbody></table></div>



Visualise the number of connection between K-mers at each position


```python
dna.plot_kmer_graph(dna.kmer_graph(sequences, kmer_size=2))
```




    <Axes: xlabel='K-mer start position, relative to methylation site', ylabel='Mean connections'>




    
![png](README_files/README_8_1.png)
    



## Documentation [Not yet implemented]

For detailed documentation and examples of all available functionalities in Nanomotif, please refer to the [official documentation](https://nanomotif-docs/docs). It provides comprehensive information on the various classes, methods, and parameters, along with usage examples and explanations.

## Contributing

We welcome contributions to Nanomotif! If you encounter any issues, have suggestions for improvements, or would like to add new features, please open an issue or submit a pull request on the [Nanomotif GitHub repository](https://github.com/SorenHeidelbach/nanomotif). We appreciate your feedback and contributions to make Nanomotif even better.

## License

Nanomotif is released under the [MIT License](https://github.com/your-username/nanomotif/blob/main/LICENSE). Feel free to use, modify, and distribute the package in accordance with the terms of the license.

## Acknowledgments

Nanomotif builds upon various open-source libraries and tools that are instrumental in its functionality. We would like to express our gratitude to the developers and contributors of these projects for their valuable work.




