# DVCS_analysis

A C++ project for Deeply Virtual Compton Scattering (DVCS) analysis built on top of the
[PARTONS](http://partons.cea.fr) framework and [libtorch](https://pytorch.org/cppdocs/) (the C++ PyTorch API).

The project implements two complementary approaches to extracting Compton Form Factors (CFFs)
from DVCS observables:

1. **PARTONS-based pipeline** — standard observable calculation using parametric GPD models and
   the PARTONS module system.
2. **Neural-network fit** — a libtorch neural network trained to predict CFFs from kinematics,
   with a fully differentiable path from NN weights to the DVCS observable.

---

## Physics background

DVCS is the process `ep → e'p'γ` in which a virtual photon scatters off a nucleon and a real
photon is produced.  The cross-section receives contributions from three amplitudes: the
Bethe-Heitler (BH) process (dominant at low beam energy), the pure DVCS amplitude, and their
interference.  The relevant non-perturbative objects are the **Compton Form Factors** (CFFs)
H, E, H̃, Ẽ — complex-valued functions of Bjorken-x (xB), momentum transfer (t), and
photon virtuality (Q²) — which are related to Generalised Parton Distributions (GPDs) by
a perturbative QCD convolution.

The primary observable studied here is the **beam-spin asymmetry**

```
A_LU^{sin1φ} = (1/π) ∫₀^{2π} dφ sin(φ) [σ(λ=+1,φ) − σ(λ=−1,φ)] / [σ(λ=+1,φ) + σ(λ=−1,φ)]
```

which is sensitive mainly to Im(H) at leading twist.  The cross-section formulae follow the
BMJ12 convention (Belitsky, Mueller, Ji — arXiv:1212.6674).

---

## Repository structure

```
DVCS_analysis/
├── src/                    # All source files (see below)
│   └── NNFit/              # Neural-network fit subsystem
│       ├── theory/         # Differentiable physics (libtorch)
│       ├── CFF_NN_Fit.cpp
│       ├── DVCSCFFNNPytorch.cpp
│       └── NN_Fit.cpp
├── include/                # Headers mirroring src/
├── bin/                    # Compiled executables (CMake output)
├── My_Analysis/
│   └── Partons_output/     # CSV output files from fits and predictions
├── libtorch/               # Bundled libtorch installation
├── cmake/Modules/          # Find-modules for external libraries
├── CMakeLists.txt
├── CLAUDE.md               # Developer notes for Claude Code
└── README.md               # This file
```

---

## Source files (in order of creation)

### Initial commit (2026-03-12)

#### `src/main.cpp` — PARTONS XML scenario runner
Entry point for `DVCS_analysis` executable.  Initialises the PARTONS singleton and dispatches
to helper functions defined in `examples.cpp` and `Compute_obs.cpp`.  Used for exploratory
calculations driven by PARTONS XML scenario files.

#### `src/examples.cpp` — PARTONS usage examples
Collection of standalone functions demonstrating the PARTONS API: computing collinear
distributions via LHAPDF, evaluating GPDs with the GK16/GK19 parametrisations, computing DVCS
and DVMP observables, and running CFF convolutions.  Covers both single-kinematic and
many-kinematic evaluations.  Serves as a reference for how to wire up the PARTONS module
system.

#### `src/Compute_obs.cpp` — DVCS observable computation over many kinematics
Implements `ComputeManyKinematicsForDVCSObservable_BSA()`, which reads a list of kinematic
points from a CSV file and uses the PARTONS service to evaluate `DVCSAluMinusSin1Phi` using
the GK16 GPD model and the GV08 process module.  Results are written to an output CSV.  This
was the first custom observable calculation in the project.

#### `src/ObsCalc_CFFNNReplicas.cpp` — Observable from PARTONS replica NN (standalone executable)
Entry point for `ObsCalc_CFFNNReplicas` executable.  Uses the built-in PARTONS `DVCSCFFNN`
module, which stores 100 pre-trained neural-network replicas (NumA++ framework, weights
hardcoded in `DVCSCFFNNReplicas.h`).  The NN takes `(log₁₀ξ, t, log₁₀Q²)` as input and
predicts `ξ·Re(CFF)` and `ξ·Im(CFF)`; the output is divided by ξ to recover the CFF.

Two analysis functions are provided:
- `analysisANN_SingleKin()` — evaluates `DVCSCrossSectionUUMinusPhiIntegrated` at one
  kinematic point across all replicas and reports a 68% confidence interval.
- `analysisANN_ManyKin()` — loops over all kinematic points from the CLAS15 BSA dataset,
  accumulates per-replica results, removes outliers (3σ cut, applied recursively), and writes
  per-point mean ± σ to `dvcs_DVCSAluSinPhi_BSACLAS15_ANN.csv`.  Individual replica values
  for the 4th kinematic point (`j==3`) are written separately to
  `dvcs_DVCSAluSinPhi_ANN_replicas.csv` for diagnostic purposes.

#### `src/dcgan.cpp` — libtorch smoke test
Minimal DCGAN implementation from the official PyTorch C++ tutorial.  Used to verify that the
libtorch installation is functional and that the CMake build links correctly against it.  Not
related to DVCS physics.

---

### Second commit (2026-04-03)

#### `src/nn_bsa.cpp` — Standalone libtorch NN trained directly on BSA data
A self-contained prototype that trains a small fully-connected network
(`5 inputs → 6 hidden (ReLU) → 1 output`) to predict `A_LU` directly from the kinematic
variables `(xB, t, Q², E, φ)`.  Input features are min-max scaled.  Reads from the CLAS07
BSA dataset (pipe-separated CSV).  Trains with Adam, evaluates MSE and R² on a held-out
validation set, and writes predictions to CSV.

This is an early experiment fitting the observable directly without any physics structure;
it served as the prototype for the more principled CFF-level fit that followed.

#### `src/NN_CFF_fit.cpp` — Entry point for NN fit to CFFs (first version)
Entry point for `NN_CFF_fit` executable.  Constructs an `NN_Fitter` (defined in
`include/NNFit/NN_Fit.h`) trained to predict a single CFF component (e.g. `ImH`) from
`(xB, t, Q²)`.  Calls `train_nn()` and `predict()`.  Uses the CLAS07 dataset.  This was the
first attempt at a CFF-level fit with a PARTONS-aware architecture but without an observable
calculation step.

#### `src/NNFit/NN_Fit.cpp` — NN fitter (first version, CFF targets)
Implements `NN_Fitter`: loads pipe-separated data, builds a `3 → 6(Tanh) → n` network
(`CFFNNModel`), trains with Adam + early stopping, and writes learning curves and predictions
to CSV.  Target labels are CFF values extracted from the data file by column name.

#### `src/Run_CFF_NN_Fit.cpp` — Entry point for the full differentiable pipeline
Entry point for `Run_CFF_NN_Fit` executable.  Constructs a `CFF_NN_Fitter` with the CLAS15
BSA dataset, runs the full four-step workflow:
1. `train_nn()` — train NN on CFF labels
2. `predict()` — evaluate on all data, compute MSE and R²
3. `observ_calc()` — compute `DVCSAluMinusSin1Phi` via PARTONS using the trained NN
4. `observ_calc_torch()` — compute the same observable via the differentiable torch pipeline

#### `src/NNFit/CFF_NN_Fit.cpp` — CFF fitter with full observable pipeline
Implements `CFF_NN_Fitter`, the central class of the NNFit subsystem:

- **`train_nn()`** — Adam (lr=1e-4, weight_decay=1e-3), min-max input scaling, early stopping
  (patience=200, max 10000 epochs), writes `cff_learning_curve.csv`.
- **`predict()`** — inference on all data, writes `cff_prediction.csv` and
  `cff_predict_model_eval.csv` (MSE + R² per CFF output).
- **`observ_calc()`** — plugs `m_net` into the PARTONS pipeline via `DVCSCFFNNPytorch`,
  uses `DVCSProcessBMJ12` + `DVCSScalesQ2Multiplier` (μF²=μR²=Q²) and calls the PARTONS
  observable service to compute `DVCSAluMinusSin1Phi`.
- **`observ_calc_torch()`** — computes the same observable using the fully differentiable
  `Theory::DVCSAluMinusSin1PhiTorch`, keeping the entire calculation inside the libtorch
  autograd graph.

#### `src/NNFit/DVCSCFFNNPytorch.cpp` — PARTONS CFF module wrapping a libtorch model
A PARTONS `DVCSConvolCoeffFunctionModule` that bridges the libtorch NN and the PARTONS
service.  Registered via `BaseObjectRegistry` at startup.  When PARTONS requests a CFF value
for a given kinematic point, `computeCFF()`:
1. Converts PARTONS skewness to Bjorken-x: `xB = 2ξ/(1+ξ)`
2. Builds input tensor `[xB, t, Q²]` and runs `m_net->forward()`
3. Reads Re/Im outputs by label and returns `std::complex<double>`

The NN outputs are treated as CFFs directly (no division by ξ), consistent with how the
training targets are defined.

---

### Added after second commit (untracked, 2026-04-24)

#### `src/NNFit/theory/DVCSKinematicsTorch.cpp` — Kinematic precomputation (BMJ12)
Pure-C++ (no torch) computation of all φ-independent kinematic quantities for a given
`(xB, t, Q², E)`: ε, K, K̃, lepton propagator decomposition, dipole electromagnetic form
factors F1/F2, BH Fourier coefficients, and the full 3×3×4 angular coefficient arrays
`C_ang` and `S_ang` for the BH-DVCS interference term following BMJ12 (arXiv:1212.6674).
All results are stored in the `DVCSKin` struct and computed once per kinematic point.

#### `src/NNFit/theory/DVCSAmplitudesBMJ12Torch.cpp` — DVCS cross-section in libtorch
Torch-tensor implementation of the BMJ12 cross-section.  CFF inputs are 0-d tensors
connected to the autograd graph; all arithmetic stays in-graph.  Provides:
- `computeDressedCFFs()` — helicity combinations F̂_X(j) for j=0,1,2
- `computeVCSCoeffs()` — purely real VCS Fourier coefficients (for unpolarised target)
- `computeInterfCoeffs()` — BH-DVCS interference coefficients linear in Re/Im(CFFs)
- `crossSectionAtPhi()` — full cross-section at a given φ and beam helicity as a 0-d tensor

#### `src/NNFit/theory/DVCSAluMinusSin1PhiTorch.cpp` — Differentiable beam-spin asymmetry
Top-level class computing `A_LU^{sin1φ}` entirely within libtorch.  Runs the NN once per
kinematic point to obtain CFF tensors, then integrates `A_LU^-(φ)` over φ ∈ [0, 2π] using
10-point Gauss–Legendre quadrature (nodes/weights from `scipy.special.p_roots(10)`,
hardcoded).  Returns a 0-d `torch::Tensor` carrying gradients w.r.t. all NN weights,
enabling direct training on observable data without a PARTONS call.

---

## Build

```bash
cd build && cmake .. && make -j$(nproc)
```

Single target:
```bash
cd build && make Run_CFF_NN_Fit
```

Executables are placed in `bin/`.  **PARTONS must be run from `bin/`**:
```bash
cd bin && ./Run_CFF_NN_Fit
```

---

## Dependencies

| Library | Purpose |
|---|---|
| PARTONS | DVCS observable and GPD calculation framework |
| ElementaryUtils | Logging, parameter handling (PARTONS dependency) |
| NumA++ | Neural network primitives used by PARTONS replicas |
| libtorch | C++ PyTorch — neural network training and autograd |
| GSL | Numerical integration (used by PARTONS) |
| LHAPDF | Parton distribution functions |
| Apfel++ | DGLAP evolution |
| libxml2 | PARTONS XML scenario parsing |

libtorch is bundled locally at `libtorch/`.  All others are found via `cmake/Modules/`.

---

## Output files (`My_Analysis/Partons_output/`)

| File | Content |
|---|---|
| `cff_learning_curve.csv` | epoch, train_loss, val_loss (every 2 epochs) |
| `cff_prediction.csv` | True vs predicted CFF values for all data points |
| `cff_predict_model_eval.csv` | MSE and R² per CFF output |
| `dvcs_DVCSAluSinPhi_BSACLAS15_ANN.csv` | Mean ± σ observable per kinematic point (replica ensemble) |
| `dvcs_DVCSAluSinPhi_ANN_replicas.csv` | Individual replica values at kinematic point j=3 |

---

## Data format

Input files are pipe-separated (`|`).  Columns: `xB | t | Q2 | ... | CFF values ... | error`.
Features are the first 3 columns `(xB, t, Q²)`; CFF labels are matched by column name from
the `output_layer` argument; the last column (`error`) is read as σ but not yet used in the
loss function.