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
│       ├── CFF_NN_Fit.cpp
│       ├── NN_Fit.cpp
│       └── theory/         # Differentiable physics layer (libtorch + PARTONS subclasses)
│           ├── Beans/Obs/DVCS/         # DVCSKinematicsTorch.cpp
│           └── Modules/                # PARTONS-registered tensor modules
│               ├── CFFs/DVCS/          # DVCSCFFNNPytorch.cpp
│               ├── Processes/DVCS/     # DVCSAmplitudesBMJ12Torch.cpp,
│               │                       # DVCSProcessBMJ12Torch.cpp
│               └── Obs/DVCS/           # DVCSAluMinusSin1PhiTorch.cpp
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

The `NNFit/theory/` layout mirrors PARTONS' own `Beans/` and `Modules/` directory hierarchy, with `CFFs`, `Processes`, and `Obs` subdivisions matching the three links of the DVCS calculation chain.

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
BSA dataset, runs the workflow:
1. `train_nn()` — train NN with χ² loss on the observable (via `CustomLoss`)
2. `observ_calc()` — compute `DVCSAluMinusSin1Phi` via the PARTONS service using the trained NN
3. `observ_calc_torch()` — compute the same observable through the PARTONS-tensor module chain (direct `computeTensor`)
4. `observ_calc_torch_via_service()` — drive the `*Torch` subclasses through the PARTONS service (verification path)

#### `src/NNFit/CFF_NN_Fit.cpp` — CFF fitter with full observable pipeline
Implements `CFF_NN_Fitter`, the central class of the NNFit subsystem:

- **`train_nn()`** — Adam (lr=1e-4, weight_decay=1e-3), raw (unscaled) inputs, early stopping
  (patience=200, max 10000 epochs), writes `cff_learning_curve.csv`. Drives observable
  training via `CustomLoss` (χ² between predicted A_LU and the data observable).
- **`observ_calc()`** — plugs `m_net` into the PARTONS pipeline via `DVCSCFFNNPytorch`,
  uses `DVCSProcessBMJ12` + `DVCSScalesQ2Multiplier` (μF²=μR²=Q²) and calls the PARTONS
  observable service to compute `DVCSAluMinusSin1Phi`.
- **`observ_calc_torch()`** — computes the same observable through the PARTONS-registered
  *tensor* module chain (`DVCSCFFNNPytorch` → `DVCSProcessBMJ12Torch` →
  `DVCSAluMinusSin1PhiTorch`), keeping the entire calculation inside the libtorch autograd
  graph. Wired through `BaseObjectRegistry` exactly like `observ_calc()`, only the three
  `*Torch` subclasses are substituted in. See the 2026-05-20 section below for details.

#### `src/NNFit/theory/Modules/CFFs/DVCS/DVCSCFFNNPytorch.cpp` — PARTONS CFF module wrapping a libtorch model
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

#### `src/NNFit/theory/Beans/Obs/DVCS/DVCSKinematicsTorch.cpp` — Kinematic precomputation (BMJ12)
Pure-C++ (no torch) computation of all φ-independent kinematic quantities for a given
`(xB, t, Q², E)`: ε, K, K̃, lepton propagator decomposition, dipole electromagnetic form
factors F1/F2, BH Fourier coefficients, and the full 3×3×4 angular coefficient arrays
`C_ang` and `S_ang` for the BH-DVCS interference term following BMJ12 (arXiv:1212.6674).
All results are stored in the `DVCSKin` struct and computed once per kinematic point.

#### `src/NNFit/theory/Modules/Processes/DVCS/DVCSAmplitudesBMJ12Torch.cpp` — DVCS cross-section in libtorch
Torch-tensor implementation of the BMJ12 cross-section.  CFF inputs are 0-d tensors
connected to the autograd graph; all arithmetic stays in-graph.  Provides:
- `computeDressedCFFs()` — helicity combinations F̂_X(j) for j=0,1,2
- `computeVCSCoeffs()` — purely real VCS Fourier coefficients (for unpolarised target)
- `computeInterfCoeffs()` — BH-DVCS interference coefficients linear in Re/Im(CFFs)
- `crossSectionAtPhi()` — full cross-section at a given φ and beam helicity as a 0-d tensor

These are pure-physics free functions in the `Theory::` namespace, independent of PARTONS.
They are wrapped by the PARTONS-registered `DVCSProcessBMJ12Torch` introduced in the next
session.

---

### Added after second commit (untracked, 2026-05-20)

This session reworked the differentiable observable pipeline so that every link of the
chain (CFFs → cross-section → asymmetry) lives **inside** a PARTONS-registered module,
each exposing a tensor-returning sibling method alongside the inherited scalar PARTONS
API. The two pipelines stay in sync by construction and can both be driven through
`BaseObjectRegistry`/`ModuleObjectFactory`. The previous standalone
`Theory::DVCSAluMinusSin1PhiTorch` driver (which bypassed PARTONS entirely) was retired in
favour of the new PARTONS subclass at the same file path.

#### `src/NNFit/theory/Modules/CFFs/DVCS/DVCSCFFNNPytorch.cpp` — `computeCFFTensor()` added
Single source of truth for the NN forward pass.
- New method:
  `std::pair<torch::Tensor, torch::Tensor> computeCFFTensor(PARTONS::GPDType::Type)`
  returns the (Re, Im) 0-d tensors for the requested CFF, with the autograd graph
  preserved (no `NoGradGuard`, no `eval()` toggle — caller controls mode).
- Existing `computeCFF()` is now a thin wrapper around `computeCFFTensor()`:
  `NoGradGuard` + `eval()` + `computeCFFTensor(m_currentGPDComputeType)` +
  `.item<float>()` → `std::complex<double>` for PARTONS.

#### `src/NNFit/theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.cpp` — new PARTONS process module
PARTONS-registered subclass of `DVCSProcessBMJ12` that exposes the cross-section in tensor
form.
- New entry point:
  `torch::Tensor crossSectionAtPhiTensor(double phi, double beamHelicity)` returns the
  total σ(λ,φ) = σ_BH + σ_VCS + σ_Interf at a single azimuth as a 0-d tensor.
- Internally `dynamic_cast`s the attached CFF module to `DVCSCFFNNPytorch*`, retrieves the
  eight leading-twist CFF tensors via `computeCFFTensor(type)` (one call per H, E, Ht, Et),
  and chains `Theory::computeDressedCFFs / computeVCSCoeffs / computeInterfCoeffs /
  crossSectionAtPhi` from `DVCSAmplitudesBMJ12Torch`.
- `buildTorchKinematics()` lazily constructs the `Theory::DVCSKin` struct from the
  inherited (`m_xB`, `m_t`, `m_Q2`, `m_E`) and caches it so a φ-scan doesn't redo the
  kinematic setup.
- All inherited scalar methods (`CrossSectionBH`, `CrossSectionVCS`, `CrossSectionInterf`)
  continue to work — PARTONS can drive this module through its normal scalar pipeline.

#### `src/NNFit/theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.cpp` — new PARTONS observable module
Replaces the previous standalone `Theory::DVCSAluMinusSin1PhiTorch`. PARTONS-registered
subclass of `PARTONS::DVCSAluMinusSin1Phi`.
- New entry point:
  `torch::Tensor computeTensor(const PARTONS::DVCSObservableKinematic&)` returns
  A_LU^{sin1φ} as a 0-d tensor carrying gradients ∂A_LU/∂(NN weights) all the way back
  through the cross-section and CFF modules to the NN parameters.
- Implementation: triggers the parent's scalar `compute()` once at the start to push
  (xB, t, Q², E) onto the process module (return value discarded), then runs a 10-point
  Gauss–Legendre quadrature, calling `DVCSProcessBMJ12Torch::crossSectionAtPhiTensor()` at
  each φ node for both helicities. GL nodes/weights (from `scipy.special.p_roots(10)`)
  moved into this class as `static const` arrays.

#### Updated `CFF_NN_Fitter::observ_calc_torch()` in `CFF_NN_Fit.cpp`
Now mirrors `observ_calc()` exactly, using the PARTONS factory pattern: all three `*Torch`
modules are instantiated via `getModuleObjectFactory()`, wired up with `setProcessModule`
/ `setConvolCoeffFunctionModule`, and driven by `pTorchObs->computeTensor(kinematic)` to
get the final 0-d tensor (printed via `.item<double>()`).

### Module chain after this refactor

```
DVCSAluMinusSin1PhiTorch    (computeTensor)            — 10-pt GL quadrature over φ
        ↓ m_pProcessModule
DVCSProcessBMJ12Torch       (crossSectionAtPhiTensor)  — σ(λ,φ) as 0-d tensor
        ↓ m_pConvolCoeffFunctionModule
DVCSCFFNNPytorch            (computeCFFTensor)         — NN CFFs as 0-d tensors
```

Each module is PARTONS-registered; PARTONS still dispatches through the inherited scalar
methods on every link. The autograd graph survives end-to-end on the tensor sibling path.
The hard boundary (`DVCSObservableService::computeSingleKinematic()`'s scalar return type)
is reached *only* by the scalar path; the differentiable path bypasses the service and
calls `computeTensor()` directly on the registered observable subclass.

### Numerical cross-check

Both pipelines evaluated at the same kinematics (xB=0.2, t=-0.2, Q²=2, E=5.932):
- PARTONS scalar `observ_calc()`:        **-0.00131307**
- PARTONS-tensor `observ_calc_torch()`:  **-0.00131306**

The ~1-in-6th-sig-fig difference is the quadrature method (PARTONS' adaptive
`MathIntegratorModule` vs. fixed 10-point Gauss–Legendre on the tensor path), not a
physics difference.

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
| `dvcs_DVCSAluSinPhi_BSACLAS15_ANN.csv` | Mean ± σ observable per kinematic point (replica ensemble, from `ObsCalc_CFFNNReplicas`) |
| `dvcs_DVCSAluSinPhi_ANN_replicas.csv` | Individual replica values at kinematic point j=3 (from `ObsCalc_CFFNNReplicas`) |

---

## Data format

Input files are pipe-separated (`|`).  Columns: `xB | t | Q2 | ... | CFF values ... | error`.
Features are the first 3 columns `(xB, t, Q²)`; CFF labels are matched by column name from
the `output_layer` argument; the last column (`error`) is read as σ but not yet used in the
loss function.