[![Paper](https://img.shields.io/badge/paper-arXiv%3A2610._-B31B1B.svg)](https://arxiv.org/abs/2610._)

# Power-Law Correlations Without Apparent Temperature-Tuned Criticality in Fixed-Tree Hierarchical Languages

Marcel Kühn, Bernd Rosenow

[arXiv:2610._](https://arxiv.org/abs/2610._)

### Abstract
Sampling temperature can substantially reshape autoregressive generation, but it is unclear when temperature changes in long-range correlations reflect a genuine phase transition rather than a smooth change in statistics. We study this question in a controlled analogue of temperature sampling: a finite-depth probabilistic context-free grammar on a fixed hierarchical tree, where exact next-token conditionals are computed by belief propagation and then rescaled by a softmax temperature before autoregressive sampling. We measure two-point mutual information between tokens as a function of separation. Across the tested temperatures and grammar ensembles, mutual information remains well described by a power-law envelope. Temperature changes the correlation magnitude and fitted exponent, but within the accessible system sizes we find no long-distance plateau characteristic of an ordered phase and no clear crossover to short-range exponential decay.
An approximate mapping to an Ising model on a tree explains how exponential decay in tree distance becomes a power law in token distance, allowing power laws to persist without tuning to a special decoding temperature. This controlled baseline clarifies which temperature effects can arise from hierarchical structure alone, before adding training, approximation error, or neural architecture.

### File Overview

* **`mi_calc.py`**: The primary script used to calculate two-point mutual information as a function of token separation across various sampling temperatures.
* **`util.py`**: Contains the core algorithmic implementations, including belief propagation logic, autoregressive sampling, and plotting routines (optimized with `numba`).
* **`rule00_plot.py` & `rule01_plot.py`**: Example scripts to produce mutual information and power-law exponent plots for the different example rule sets.
* **`example_use.ipynb`**: A Jupyter notebook demonstrating how to run the pipeline and visualize the results. **Note:** The hyperparameters defined in this notebook are meant purely as a fast, illustrative example and do not correspond to the actual parameters used to generate the final figures in the paper.

### Dependencies
Running the code requires standard scientific Python libraries, notably `numpy`, `scipy`, `matplotlib`, and `numba`.
