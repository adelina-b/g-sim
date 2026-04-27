# $\mathfrak{g}$-sim: Fast classical simulation of structured quantum circuits

This repository accompanies the paper

- A. Bärligea et al., "Enabling Lie Algebraic Classical Simulation beyond Free Fermions" — <https://arxiv.org/abs/2604.16701>

and builds on the Lie-algebraic simulation framework of

- M. L. Goh et al., “Lie-algebraic classical simulations for quantum computing” — <https://doi.org/10.1103/3y65-f5w6>

The goal of this repository is twofold:

1. to provide the code used to reproduce the numerical experiments of the accompanying paper, and  
2. to expose the optimized basis-specific preprocessing primitives that underlie these simulations.

At its current stage, the repository contains a high-performance implementation of several symmetry-adapted preprocessing backends, together with experiment-specific scripts and tutorials showing how to turn them into full $\mathfrak{g}$-sim workflows. A more unified and general implementation of $\mathfrak{g}$-sim, covering the full pipeline from preprocessing to simulation in a single interface, is currently under development.

## What is implemented here

The core idea of $\mathfrak{g}$-sim is to simulate quantum dynamics in an operator basis adapted to the dynamical Lie algebra of the circuit, rather than in the full Hilbert space. When the relevant Lie algebra or invariant operator subspace has only polynomial dimension in the system size, this can lead to efficient classical simulation of expectation values, gradients, and related quantities.

This repository currently contains optimized preprocessing backends for four basis types:

- Pauli strings  
- (translational-invariant) Pauli cycles  
- (permutation-invariant) Pauli orbits  
- the MGGM basis for fixed-Hamming-weight sectors

These primitives are then used in the experiment scripts in `examples/` to build complete simulation pipelines for the numerical case studies in the paper.

## Repository structure

```text
g-sim/
├── examples/
│   ├── data/
│   │   ├── adj_gcycles_TFIM_n50.npz
│   │   ├── adj_gmggm_k2_n10.npz
│   │   ├── adj_gorbits_targeted_n50.npz
│   │   └── adj_gstrings_TFIM_n50.npz
│   ├── hwencoder_demonstration.ipynb
│   ├── hwencoder_helpers.py
│   ├── hwencoder_simulator.py
│   ├── peqnn_demonstration.ipynb
│   ├── peqnn_helpers.py
│   ├── peqnn_simulator.py
│   ├── tfim_demonstration.ipynb
│   ├── tfim_helpers.py
│   └── tfim_simulator.py
├── src/
│   └── gsim/
│       ├── __init__.py
│       ├── gcycles.py
│       ├── gmggm.py
│       ├── gorbits.py
│       ├── gstrings.py
│       └── utils.py
├── pyproject.toml
└── README.md
````

## The `gsim` package

The actual Python package lives in `src/gsim`. It contains the low-level preprocessing routines used to construct Lie bases and sparse commutator / generator data in different symmetry-adapted representations.

### `gstrings.py`

Pauli-string preprocessing.

This module provides Lie closure and preprocessing routines when generators are given explicitly as Pauli strings or as sums of Pauli strings. It is the most direct backend and is used in the free-fermionic TFIM example.

### `gcycles.py`

Pauli-cycle preprocessing for Translation-invariant systems.

This backend supports one-dimensional translational settings, including both periodic and open boundary conditions. Internally, it distinguishes canonical cyclic representatives from open-boundary bulk/fixed decompositions.

### `gorbits.py`

Pauli-orbit preprocessing for permutation-invariant systems.

This module implements preprocessing in the orbit basis labeled by triples ((p,q,r)), counting the number of (X), (Y), and (Z) factors in an orbit. It supports both full structure-constant computation and targeted construction of generator matrices for selected orbit generators.

### `gmggm.py`

MGGM preprocessing for fixed-Hamming-weight sectors.

This module provides backends for simulations restricted to the fixed-Hamming-weight subspace (\mathcal H_k). It includes both full preprocessing and a targeted implementation specialized to the antisymmetric generators relevant for RBS-based state-preparation circuits.

### `utils.py`

Shared utilities.

This file contains helper routines used across the package, including loading / saving sparse preprocessing data and basis-related helpers.

## The `examples/` directory

The `examples/` directory contains the code corresponding to the numerical experiments in the paper. These scripts are intentionally problem-specific: they illustrate how the basis primitives in `src/gsim` can be assembled into complete simulation workflows for concrete structured quantum circuits.

At present, the examples are not meant to be a polished general API. Instead, they are included to make the numerical section of the paper directly reproducible and to serve as worked tutorials for constructing $\mathfrak{g}$-sim pipelines from the primitive backends.

### Current examples

#### `tfim_*`

Free-fermionic TFIM / TFIM-HVA example based on Pauli-string preprocessing.

These files show how to:

* construct the relevant Pauli-string generators,
* build or load the corresponding adjoint block data,
* evaluate TFIM expectation values and gradients efficiently in adjoint space,
* compare against a direct circuit simulator in the accompanying notebook.

#### `peqnn_*`

Permutation-equivariant QNN example for graph-state classification.

These files show how to:

* construct the Pauli-orbit representation of graph-state inputs,
* build the permutation-invariant loss function,
* evaluate the loss and its gradients via targeted orbit-generator matrices,
* validate the implementation against a direct reference simulation.

#### `hwencoder_*`

Fixed-Hamming-weight amplitude encoding example.

These files implement the HW-preserving encoder used in the paper’s state-preparation experiment. They demonstrate how to:

* construct the Chase ordering of fixed-Hamming-weight basis states,
* map controlled RBS gates to MGGM generators,
* simulate the encoding entirely in the MGGM basis,
* compare the encoded distribution to the target distribution in the notebook tutorial.

### `examples/data/`

This folder contains precomputed sparse preprocessing data used by some of the examples, such as targeted generator matrices or adjoint block decompositions. These files are included to make the demonstrations easier to run and to avoid repeating expensive preprocessing when reproducing the experiments.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/adelina-b/g-sim.git
cd g-sim
```

### 2. Install the package in editable mode

```bash
python -m pip install -r requirements.txt  
python -m pip install -e .
```

After this, the package can be imported as

```python
import gsim
```

or, for example,

```python
from gsim import gstrings, gorbits, gmggm, gcycles
```

## PauliEngine dependency

For Pauli-string preprocessing in the symplectic representation, the current implementation depends on `PauliEngine`.

Please install `PauliEngine` separately by following its installation instructions:

* `[https://github.com/tequilahub/pauliengine]`

At the moment, users have to rebuild `PauliEngine` for their own system and Python version.
This is a temporary limitation of the current development state; a cleaner installation route is planned for future releases.

## Running the tutorials

The easiest way to get started is to open one of the demonstration notebooks in `examples/`:

* `examples/tfim_demonstration.ipynb`
* `examples/peqnn_demonstration.ipynb`
* `examples/hwencoder_demonstration.ipynb`

These notebooks are meant to be instructive entry points. They explain the relevant simulation task, show how the preprocessing data are used, and compare the $\mathfrak{g}$-sim outputs against direct reference simulations whenever feasible.

## Current status

This repository is under active development.

What is already included:

* optimized basis-specific preprocessing routines,
* the code needed to reproduce the numerical experiments of the accompanying paper,
* experiment notebooks demonstrating the use of the implemented primitives.

What is still in progress:

* a more unified end-to-end $\mathfrak{g}$-sim interface,
* broader support for reusable high-level simulator APIs,
* cleaner installation of dependencies,
* improved serialization and loading of preprocessing data,
* more general and better documented workflows beyond the paper-specific examples.

In other words, the present repository should be viewed as:

* a reproducibility repository for the paper, and
* an intermediate public release of the core preprocessing technology,

rather than the final form of the software framework.

## Reproducing the paper

The numerical examples reported in the paper are implemented in the `examples/` directory. The corresponding scripts and notebooks demonstrate the three main case studies:

* free-fermionic TFIM dynamics,
* permutation-equivariant quantum graph classification,
* fixed-Hamming-weight amplitude encoding.

Because these workflows are currently tailored to the specific experiments in the manuscript, some paths, parameterizations, and preprocessing assumptions are intentionally specialized. They nevertheless provide fully worked examples of how the underlying basis primitives are used in practice.

## Citation

If you use this repository, please cite:

```bibtex
@misc{Barligea2026},
      title={Enabling Lie-Algebraic Classical Simulation beyond Free Fermions}, 
      author={Adelina Bärligea and Matthew L. Sims-Goh and Jakob S. Kottmann},
      year={2026},
      eprint={2604.16701},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.16701}, 
}
```

and, where appropriate, also cite the original $\mathfrak{g}$-sim paper:

```bibtex
@article{Goh2025,
  title = {Lie-algebraic classical simulations for quantum computing},
  volume = {7},
  ISSN = {2643-1564},
  url = {http://dx.doi.org/10.1103/3y65-f5w6},
  DOI = {10.1103/3y65-f5w6},
  number = {3},
  journal = {Physical Review Research},
  publisher = {American Physical Society (APS)},
  author = {Goh,  Matthew L. and Larocca,  Martin and Cincio,  Lukasz and Cerezo,  M. and Sauvage,  Frédéric},
  year = {2025},
  month = sept 
}
```

## License

This repository is released under the MIT License. See the [LICENSE](LICENSE) file for the full license text.

