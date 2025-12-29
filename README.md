# qnoe-3ChTMM
Python package implementing a three-channel Transfer Matrix Method (TMM) to model scattering processes between one-dimensional guided modes and a localized scattering element. The framework combines standard TMM propagation in the near-field channel with a symmetry-constrained 3×3 scattering matrix describing coupling to a far-field channel. It is designed for efficient and physically interpretable simulations of 1D guided-mode scattering problems, with particular relevance to near-field nanophotonics and polaritonic systems.

## Installation
1. Clone the repository to your hard-drive
2. Open the cmd
3. Install the package by typing: 
	pip install -e "~\qnoe-3ChTMM"

## Tutorial and Examples

The `examples/` folder contains Python scripts demonstrating how to use `qnoe-3ChTMM` for guided-mode propagation, scattering, and near-field simulations in one-dimensional polaritonic systems.

### Example 01 — Three-channel TMM and coupling effects
This example introduces the three-channel Transfer Matrix Method framework. It:
- Computes graphene plasmon dispersion relations
- Builds a Fabry–Pérot cavity using effective interfaces and chunks
- Compares standard two-channel TMM with the three-channel model
- Demonstrates the effect of coupling strength and coupling phase of the 3-port scattering device on near-field and far-field observables

This example illustrates how the localized scatterer perturbs an otherwise unperturbed guided-mode system.

---

### Example 02 — hBN phonon polaritons and near-field fringes
This example simulates hyperbolic phonon polaritons in hexagonal boron nitride (hBN). It:
- Defines a polaritonic material using effective mode permittivity
- Models reflection at a flake edge with a prescribed phase
- Computes spatially resolved near-field interference fringes

This script demonstrates near-field scanning simulations for material boundaries.

---

### Example 03 — Graphene plasmons and near-field detection schemes
This example focuses on graphene plasmon resonators and near-field signal formation. It:
- Computes transmission using standard TMM
- Simulates near-field response using simplified s-SNOM models
- Compares pseudo-heterodyne and self-homodyne detection schemes
- Studies the dependence of near-field contrast on tip–sample coupling strength

This example highlights how detection schemes and coupling parameters affect measured near-field signals.

---

### Example 04 — Graphene plasmonic photonic lattice
This example models a one-dimensional graphene plasmonic lattice. It:
- Computes band structures of periodic systems
- Simulates reflection spectra of finite lattices
- Performs spatially and spectrally resolved near-field scans

This script demonstrates how `qnoe-3ChTMM` can be used to study polaritonic crystals and finite-size effects.

---

## License
The package qnoe-3ChTMM is distributed under the GNU Lesser General Public License v3 (LGPLv3).

## Cite Us
If you use qnoe-3ChTMM for your research please cite: https://arxiv.org/abs/2307.11512
