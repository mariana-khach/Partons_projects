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
│       ├── CFF_NN_Fit.cpp   # CFF_NN_Fitter: train / predict / replicas / verification paths
│       ├── CustomLoss.cpp   # reduced chi^2/n on the observable, through the tensor chain
│       ├── NN_Fit.cpp
│       └── Theory/         # Differentiable physics layer (libtorch + PARTONS subclasses)
│           └── Modules/                # PARTONS-registered tensor modules + generic templates
│               ├── MathIntegratorModuleTorch.cpp
│               ├── CFFs/DVCS/          # DVCSCFFModuleTorch.h (interface),
│               │                       # DVCSCFFNNTorch.cpp,
│               │                       # DVCSCFFScalarTorch.cpp (scalar-model adapter)
│               ├── Processes/          # ProcessModuleTorch.h (generic)
│               │   └── DVCS/           # DVCSProcessModuleTorch.h, DVCSProcessBMJ12Torch.cpp
│               ├── Obs/                # ObservableTorch.h (generic)
│               │   └── DVCS/           # DVCSObservableTorch.h, DVCSAluMinusTorch.cpp,
│               │                       # DVCSAluMinusSin1PhiTorch.cpp
│               └── Services/           # ObservableServiceTorch.h (generic mixin)
│                   └── DVCS/           # DVCSObservableServiceTorch.cpp
├── include/                # Headers mirroring src/
├── bin/                    # Compiled executables (CMake output)
├── My_Analysis/
│   ├── Codes/              # Analysis notebooks (learning curves, predictions, CFF scans/bands)
│   │   ├── CFF_obs_train_predict_plot.ipynb
│   │   ├── CFF_obs_train_predict_plot_xpow.ipynb
│   │   ├── CFF_plots_ALU_2007_xpow_replica.ipynb
│   │   ├── CFF_plots_ALU_2007_xpow_replica_Farm.ipynb
│   │   └── Obs_NN_train_predict_plot.ipynb
│   └── Partons_output/     # CSV/JSON output files from fits and predictions
├── libtorch/               # Bundled libtorch installation
├── cmake/Modules/          # Find-modules for external libraries
├── CMakeLists.txt
├── CLAUDE.md               # Developer notes for Claude Code
└── README.md               # This file
```

The `NNFit/Theory/` layout mirrors PARTONS' own `Modules/` directory hierarchy, with `CFFs`,
`Processes`, `Obs`, and `Services` subdivisions matching the links of the DVCS calculation
chain.  (A `Beans/Obs/DVCS/` directory existed until 2026-06-17 and is gone — see the
2026-04-24 section below.)

---

## Branches

Work lands on **`devel`**; `main` lags behind it.  As of **2026-09-22** `origin/devel` carries
everything documented here: the batched torch chain (Option A), the GL-20 φ-quadrature order, the
replica failure policy, and the CFF-link tensor interface with its scalar-model adapter.  The
`vect_optionA` feature branch was merged there.

`vectorized_calc` holds an alternative **Option B (raw-tensor)** batching experiment that was not
taken — the shipped design pushes the batch dimension down into the existing layers instead.

Note for issue tracking: the repository's default branch is `main`, and GitHub only auto-closes an
issue when the closing commit reaches the default branch — so a `closes #N` merged into `devel`
stays open until `devel` reaches `main`, and is normally closed by hand.

Dated sections below name the branch each piece of work happened on; they are kept as written
rather than updated when a branch merges.

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
  for **every** kinematic point are written separately to
  `dvcs_DVCSAluSinPhi_ANN_replicas.csv` for diagnostic purposes (originally only the 4th
  point, `j==3`; fixed on branch `adding_replicas`, 2026-07).

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
Entry point for `Run_CFF_NN_Fit` executable.  Constructs a `CFF_NN_Fitter` with the CLAS07
BSA dataset (`BSA_CLAS_07_KK_format_ALU_error.csv`, 16 points), the output-layer list (e.g.
`{"ImH"}`) and `x_pow`, then runs the workflow:
1. `train_nn()` — central fit: train the NN with a reduced-χ²/n loss on the observable (via `CustomLoss`)
2. `predict()` — evaluate the trained model's observable prediction per point; write `obs_prediction.csv`, `obs_model_eval.csv`, `cff_model.json`
3. `observ_calc()` — compute `DVCSAluMinusSin1Phi` via the PARTONS service using the trained NN (base PARTONS classes)
4. `observ_calc_torch()` — the same observable through the PARTONS-tensor module chain (`DVCSObservableServiceTorch::computeSingleKinematicTorch`, gradient preserved)
5. `observ_calc_torch_scalar()` — drive the `*Torch` subclasses through the *scalar* PARTONS service (verification path; formerly `observ_calc_torch_via_service()`)
6. `train_replicas(10)` + `export_replicas(OUT_DIR)` — Monte Carlo replica ensemble for the CFF uncertainty band

`main()` also wraps the run in a `std::chrono::steady_clock` measurement and prints
`Total run time: <s> s` at exit (replica training multiplies the single-fit cost by
`n_replicas × (1 + retries)`).

#### `src/NNFit/CFF_NN_Fit.cpp` — CFF fitter with full observable pipeline
Implements `CFF_NN_Fitter`, the central class of the NNFit subsystem:

- **`train_nn()`** — central (unsmeared) fit.  A thin wrapper over the shared `fit_once()`
  helper: Adam (lr=1e-2, **no** weight decay since 2026-08-05), raw inputs with the min-max
  scaling applied *inside* the NN module, early stopping (patience=1000, max 10000 epochs)
  with best-validation parameter restore, writes `cff_learning_curve.csv`.  Drives observable
  training via `CustomLoss` (reduced χ²/n between predicted A_LU^{sin1φ} and the measured
  observable).
- **`predict()`** — evaluates the trained model's **observable** prediction per data point
  through the same tensor chain, writes `obs_prediction.csv` (per-point true/pred),
  `obs_model_eval.csv` (MSE, R², reduced χ²/n) and `cff_model.json`.
- **`train_replicas()` / `export_replicas()`** — the Monte Carlo replica ensemble
  (see the 2026-09-01 section below).
- **`observ_calc()`** — plugs `m_net` into the PARTONS pipeline via `DVCSCFFNNTorch`,
  uses `DVCSProcessBMJ12` + `DVCSScalesQ2Multiplier` (μF²=μR²=Q²) and calls the PARTONS
  observable service to compute `DVCSAluMinusSin1Phi`.
- **`observ_calc_torch()`** — computes the same observable through the PARTONS-registered
  *tensor* module chain (`DVCSCFFNNTorch` → `DVCSProcessBMJ12Torch` →
  `DVCSAluMinusSin1PhiTorch`), keeping the entire calculation inside the libtorch autograd
  graph. Wired through `BaseObjectRegistry` exactly like `observ_calc()`, only the three
  `*Torch` subclasses are substituted in. See the 2026-05-20 section below for details.

#### `src/NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.cpp` — PARTONS CFF module wrapping a libtorch model
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

> **Removed 2026-06-17** (commit `97c81b4`).  The two files described in this section
> (`DVCSKinematicsTorch`, `DVCSAmplitudesBMJ12Torch`) were the *standalone* `Theory::`
> physics layer that bypassed PARTONS.  They were superseded by the in-framework port
> described from the 2026-06-16 section onwards, where the BMJ12 kinematics and Fourier
> coefficients live in `DVCSProcessBMJ12Torch::setupKinematicsTorchBatch`.  Kept here as
> history — neither file exists in the tree today.

#### `src/NNFit/Theory/Beans/Obs/DVCS/DVCSKinematicsTorch.cpp` — Kinematic precomputation (BMJ12)
Pure-C++ (no torch) computation of all φ-independent kinematic quantities for a given
`(xB, t, Q², E)`: ε, K, K̃, lepton propagator decomposition, dipole electromagnetic form
factors F1/F2, BH Fourier coefficients, and the full 3×3×4 angular coefficient arrays
`C_ang` and `S_ang` for the BH-DVCS interference term following BMJ12 (arXiv:1212.6674).
All results are stored in the `DVCSKin` struct and computed once per kinematic point.

#### `src/NNFit/Theory/Modules/Processes/DVCS/DVCSAmplitudesBMJ12Torch.cpp` — DVCS cross-section in libtorch
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

#### `src/NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.cpp` — `computeCFFTensor()` added
Single source of truth for the NN forward pass.
- New method:
  `std::pair<torch::Tensor, torch::Tensor> computeCFFTensor(PARTONS::GPDType::Type)`
  returns the (Re, Im) 0-d tensors for the requested CFF, with the autograd graph
  preserved (no `NoGradGuard`, no `eval()` toggle — caller controls mode).
- Existing `computeCFF()` is now a thin wrapper around `computeCFFTensor()`:
  `NoGradGuard` + `eval()` + `computeCFFTensor(m_currentGPDComputeType)` +
  `.item<float>()` → `std::complex<double>` for PARTONS.

#### `src/NNFit/Theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.cpp` — new PARTONS process module
PARTONS-registered subclass of `DVCSProcessBMJ12` that exposes the cross-section in tensor
form.
- New entry point:
  `torch::Tensor crossSectionAtPhiTensor(double phi, double beamHelicity)` returns the
  total σ(λ,φ) = σ_BH + σ_VCS + σ_Interf at a single azimuth as a 0-d tensor.
- Internally `dynamic_cast`s the attached CFF module to `DVCSCFFNNTorch*`, retrieves the
  eight leading-twist CFF tensors via `computeCFFTensor(type)` (one call per H, E, Ht, Et),
  and chains `Theory::computeDressedCFFs / computeVCSCoeffs / computeInterfCoeffs /
  crossSectionAtPhi` from `DVCSAmplitudesBMJ12Torch`.
- `buildTorchKinematics()` lazily constructs the `Theory::DVCSKin` struct from the
  inherited (`m_xB`, `m_t`, `m_Q2`, `m_E`) and caches it so a φ-scan doesn't redo the
  kinematic setup.
- All inherited scalar methods (`CrossSectionBH`, `CrossSectionVCS`, `CrossSectionInterf`)
  continue to work — PARTONS can drive this module through its normal scalar pipeline.

#### `src/NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.cpp` — new PARTONS observable module
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
DVCSCFFNNTorch            (computeCFFTensor)         — NN CFFs as 0-d tensors
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

### Torch observable chain — link-for-link mirror of the scalar path (2026-06-16)

The differentiable pipeline was reworked into a Torch chain that is **structurally
identical, link-for-link, to PARTONS' scalar chain**: every scalar link has a torch twin
with the same role, so gradients (∂A_LU/∂NN-weights) flow end-to-end while each class stays
a drop-in for the scalar pipeline.  `DVCSCFFNNPytorch` was also renamed **`DVCSCFFNNTorch`**
in this rework.

**Generic, channel-agnostic templates** (tensor twins of PARTONS' `Observable<K,R>` /
`ProcessModule<K,R>` / `ObservableService<K,R>`; `ResultType` collapses to `torch::Tensor`
(to `ObservableResultTorch<K>` as of 2026-09-23 — see that section),
so only `KinematicType` is templated):

- **`ObservableTorch<K>`** — NVI idiom mirroring scalar `compute`/`computeObservable`:
  public `computeTensor()` delegates to the protected pure-virtual `computeTensorImpl()`.
- **`ProcessModuleTorch<K>`** — channel-agnostic process skeleton.
- **`ObservableServiceTorch<K>`** — a **mixin** (not a base) adding the tensor driver
  `computeSingleKinematicTorch(kin, ObservableTorch<K>*)`, layered onto the existing PARTONS
  service.

**DVCS channel layer** (each twin sits next to its scalar counterpart).  *This is the
2026-06-16 single-kinematic form; for the current batched names see the batching section
below:*

```
ObservableServiceTorch::computeSingleKinematicTorch   ↔  ObservableService::computeSingleKinematic
ObservableTorch::computeTensor (template method)       ↔  Observable::compute
   computeTensorImpl (hook)                            ↔     computeObservable
DVCSAluMinusTorch::aLUTensor (pointwise)               ↔  DVCSAluMinus::computeObservable
DVCSProcessModuleTorch::crossSectionTensor (Σ, sel.)   ↔  DVCSProcessModule::compute(…,VCSSubProcessType)
   crossSectionBH/VCS/InterfTensor                     ↔     CrossSectionBH/VCS/Interf
   setupKinematicsTorch                                ↔     setKinematics + CFF forward
DVCSCFFNNTorch::computeAllCFFsTensor                   ↔  DVCSCFFNNTorch::computeCFF
```

`DVCSObservableServiceTorch` inherits `PARTONS::DVCSObservableService` (putting it on the
registrable `ServiceObject` branch, reusing the full scalar machinery) and mixes in
`ObservableServiceTorch<DVCSObservableKinematic>` for the tensor driver; it self-registers in
`BaseObjectRegistry` and is fetched by name.

**Dual-use (single source of truth = the tensor method).**  Each module's scalar virtual
wraps its tensor twin under `NoGradGuard` + `.item()` (`computeCFF()`→`computeCFFTensor`,
`computeObservable()`→`computeTensor`), so the *same* registered classes serve both the
differentiable path and PARTONS' ordinary scalar pipeline.  Three verification paths in
`CFF_NN_Fit.cpp` — base PARTONS scalar, `*Torch` scalar virtuals, and `*Torch` tensor entry
points — agree to every printed digit, with `requires_grad = true` preserved on the tensor
path.

> **Coverage caveat:** only the **unpolarized-target** BMJ12 sector is ported (valid for
> A_LU and siblings).  For polarized-target observables use the base PARTONS classes — a
> `*Torch` leaf's scalar path silently runs the unpolarized port.  See `CLAUDE.md`.

---

### Integration speedup — DExp → 10-point Gauss–Legendre (2026-06-22)

The φ-integration twin **`MathIntegratorModuleTorch`** (a libtorch counterpart of
`PARTONS::MathIntegratorModule`, inherited by `DVCSAluMinusSin1PhiTorch`) replaces the
hard-coded φ-loop with a configurable, **gradient-preserving** quadrature.  Every supported
rule reduces to `∫ = Σ wᵢ f(xᵢ)` with **constant** nodes/weights (no dependence on NN
parameters), so the autograd graph flows entirely through `f(xᵢ)`.

The leaf was switched from the adaptive **double-exponential (DEXP)** rule to a fixed
**10-point Gauss–Legendre (GL)** rule:

```cpp
// DVCSAluMinusSin1PhiTorch constructor
// was: MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::DEXP);
MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 10);
```

**Why:** the `A_LU^{sin1φ}` integrand is smooth and 2π-periodic, so a fixed rule needs **one
batched integrand evaluation** (all φ nodes at once) versus DEXP's adaptive multi-level
`L+1` evaluations.  The GL nodes/weights come from NumA's Gauss–Legendre rule generator
(roots of `P₁₀`, `wᵢ = 2/[(1−xᵢ²)P₁₀′(xᵢ)²]`); `MathIntegratorModuleTorch` reads them via
`getNodes()/getWeights()` and remaps onto `[0, 2π]`.

**Verification** (xB=0.2, t=−0.2, Q²=2, E=5.932):

| Path | Quadrature | A_LU^{sin1φ} |
|---|---|---|
| `observ_calc()` (base PARTONS, scalar) | DEXP adaptive | −0.00895582 |
| `observ_calc_torch()` (torch tensor path) | **GL-10** | −0.00895585 |
| `observ_calc_torch_scalar()` (torch scalar virtuals) | GL-10 | −0.00895585 |

GL-10 reproduces the DEXP value to ~6 significant figures (~3×10⁻⁸ absolute) — far below
fit/measurement precision — while collapsing the φ-integral to a single batched evaluation.
The swap is per-observable: re-validate GL vs DEXP before reusing it for a different
integrand.

---

### Speedup work completed

(#3, batching across data points, landed later — see
**"Batching across data points"** below.)

| Speedup | Status | What it does |
|---|---|---|
| **Fixed GL integrator** (was DEXP) | ✅ done (2026-06-23; order raised 10 → 20 on 2026-09-21) | One batched φ-integrand evaluation instead of DEXP's adaptive `L+1` levels (per A_LU computation) |
| **Hoist setup — #1** (prepare/assemble split) | ✅ done (2026-06-25) | The φ- and helicity-independent kinematic factors + CFFs are computed **once per kinematic point** instead of once per beam helicity (twice) |
| **Batch across data points — #3** (vectorization) | ✅ done (2026-09-15) | The per-point loop is gone: `[N]` kinematics, one `[N,3]` NN forward, `[N,M]` cross-sections, one `backward()` per epoch |
| **Multithread the per-point loop — #4** | ➖ moot | Superseded by #3 — no per-point loop remains to thread (and it would have carried shared-`.grad` races) |

All three implemented speedups are **value-preserving**: the three `observ_calc*` paths agree
to every printed digit after each of them.

---

### Hoist setup — prepare/assemble split (#1, 2026-06-24)

A standalone per-point speedup that also de-risked the batched work (#3, below).  The torch
asymmetry `DVCSAluMinusTorch::aLUTensor` evaluates the cross section for **both** beam
helicities (λ=±1) to form `A_LU = (σ⁺−σ⁻)/(σ⁺+σ⁻)`.  Previously each call re-ran the whole
φ-independent setup — **one NN forward + the BMJ12 kinematic block + 72 angular
coefficients** — so that helicity-independent work executed **twice per data point**.  (The
scalar PARTONS path has the same two-helicity structure but dodges the cost via CFF caching;
the torch path can't cache without severing the autograd graph, so it hoists instead.)

> **Superseded 2026-09-16:** the single-point API shown in this section
> (`prepareTensor`, the four `crossSectionTensor` overloads, `setupKinematicsTorch`,
> `aLUTensor`) was **deleted** once #3 landed — see the last section below.  The
> prepare/assemble *idea* survives unchanged in the batched twins
> (`prepareTensorBatch` / `crossSectionTensorBatch` / `setupKinematicsTorchBatch` /
> `aLUTensorBatch`).  The description below is kept as the rationale for that factoring.

The monolithic `crossSectionTensor` was split into **prepare + assemble** in
`DVCSProcessModuleTorch`:

- **`prepareTensor(kin)`** — runs the φ-/helicity-independent `setupKinematicsTorch` once
  (sets an `m_prepared` guard);
- **lightweight `crossSectionTensor(λ, charge, φ[, processType])`** — assumes prepare ran;
  only sums the requested sub-processes (BH/VCS/INT);
- **self-contained `crossSectionTensor(λ, charge, kin, φ[, processType])`** — kept for
  single-shot callers, now delegates (`prepareTensor` → lightweight assemble).

`aLUTensor` now calls `prepareTensor(kin)` **once**, then the lightweight overload per
helicity:

```cpp
pProc->prepareTensor(kinematic);                            // setup ONCE
auto sigmaPlus  = pProc->crossSectionTensor(+1., -1., phi); // assemble only
auto sigmaMinus = pProc->crossSectionTensor(-1., -1., phi); // assemble only
```

So the NN forward + 72-coefficient build + kinematic block run **once** per point instead of
twice.  The split changes only scheduling — the physics and the per-link chain structure are
untouched — and it is the exact prepare/assemble factoring the batched path (#3) lifts to
`[N]` tensors.  Two source files changed: `DVCSProcessModuleTorch.h`,
`DVCSAluMinusTorch.cpp`.

**Verification** (fresh-seeded NN; only cross-path agreement matters):

| Path | A_LU^{sin1φ} | grad |
|---|---|---|
| `observ_calc()` (base PARTONS, scalar) | 0.089754 | — |
| `observ_calc_torch()` (torch tensor path) | 0.0897542 | `requires_grad = true` |
| `observ_calc_torch_scalar()` (torch scalar virtuals) | 0.0897542 | — |

All three agree to every printed digit and the tensor path keeps its gradient (the single
shared CFF forward feeding both helicities gives an identical value and gradient by the chain
rule).

---

### Trained-model export + CFF scans (2026-06-29)

`CFF_NN_Fitter::predict()` now exports the trained network so the CFFs can be scanned and
plotted out-of-process:

- **`export_model_json()`** writes `cff_model.json` — `arch`, `dtype`, `input_features`,
  `output_layer` (the CFF labels = `m_output_layer`), the min-max `scaling` (`x_min`/`x_max`),
  and the `fc1`/`fc2` weights+biases (`[out, in]` orientation, `y = x Wᵀ + b`). This lets the
  exact NN forward be reproduced in Python with **numpy only** (no torch dependency):
  `tanh((x − xmin)/(xmax − xmin) @ W1ᵀ + b1) @ W2ᵀ + b2`. JSON was chosen over `.pt` because a
  hand-written C++ `nn::Module` doesn't cross cleanly to Python (except via TorchScript) and
  the net is tiny, so the numpy forward is trivial (validated to ~5×10⁻⁶ against the C++
  outputs).
- The notebook **`CFF_obs_train_predict_plot.ipynb`** gained a CFF-scan section: it loads
  `cff_model.json`, reproduces the forward, and plots **CFFs vs xB** (fixed t, Q²) and
  **CFFs vs −t** (fixed xB, Q²) with the fixed kinematics annotated, saving each to PNG. One
  line per CFF in `output_layer`.

### Integrator node caching (2026-06-29)

`MathIntegratorModuleTorch` now **caches** the fixed-rule (GL/TRAPEZOIDAL) reference
nodes/weights as `mutable` member tensors, converted from NumA once and reused across
`integrateTorch()` calls instead of rebuilt every call; only the `[a, b]` remap runs per call.
Rebuilt if the node count changes and cleared by `setIntegrator` on a rule change. A pure
caching optimization — **value-preserving** (the three `observ_calc*` paths still agree). For
GL-10 the saving is tiny; it matters more at high node counts. (DEXP already used its
program-wide static `dExpTables()`; TRAPEZOIDALLOG is inherently per-`[a, b]`.)

---

### Branch `adding_replicas` — training robustness + replica-output groundwork (2026-07)

Preparatory changes for the CFF-uncertainty replica ensemble (train N replicas on
Monte-Carlo-fluctuated data, band = mean ± σ in Python) — implemented 2026-09-01, see below —
plus training quality-of-life fixes:

- **Best-validation model snapshot in `train_nn()`** — the parameters are snapshotted
  (`p.detach().clone()`) every time the validation χ² improves and restored into the net when
  training ends (early stop *or* max-epochs). The stored model is now the one early stopping
  actually selected, not the last epoch's weights (which are up to `patience` = 200 steps past
  the optimum). The selected validation χ² is kept in `m_best_val_loss` and exported by
  `export_model_json()` as a new **`best_val_chi2`** field in `cff_model.json`.
- **Learning rate raised** — Adam lr 1e-3 → **1e-2** (weight_decay unchanged at 1e-3).
- **Live-tailable learning curve** — `cff_learning_curve.csv` is opened once and flushed after
  each write instead of reopened per logging step, so `tail -f` works during a long run.
- **Per-replica output fix in `ObsCalc_CFFNNReplicas`** — `analysisANN_ManyKin()` now writes
  the individual replica values for **every** kinematic point to
  `dvcs_DVCSAluSinPhi_ANN_replicas.csv` (previously only the 4th point, `j==3`). Closes the
  per-replica-capture task open since 2026-04-24.
- **Notebooks moved to `My_Analysis/Codes/`** — `CFF_obs_train_predict_plot.ipynb` and
  `Obs_NN_train_predict_plot.ipynb` now live in a dedicated code directory and read their
  inputs from `../Partons_output/`. The CFF notebook also reports the best validation χ²
  from `cff_model.json` (falling back to the learning-curve minimum for older exports that
  predate the `best_val_chi2` field).
- **Repo hygiene** — `.gitignore` now excludes `*.csv`, `*.png`, and `.ipynb_checkpoints/`;
  the `*_beforespeedup` snapshot CSVs and the CFF-scan PNGs were removed from tracking
  (they are regenerated outputs).

---

### CFF rescaling `CFF = xB^x_pow · NN(x)` (2026-08-05)

An optional power-law prefactor so the network can learn a rescaled target instead of the raw
CFF — motivated by CFFs (e.g. Im H) that vary over orders of magnitude near small xB, where
`xB^p · NN(x)` is easier to fit than the CFF directly.

- `DVCSCFFNNTorch::forwardNN()` returns `NN_output · xB^m_xPow`; the exponent arrives through
  `setModel(net, outputLayer, xMin, xMax, xPow)` so the model and *all* of its preprocessing
  travel together.  `x_pow = 0.0` (the default) is a no-op, so every existing call site is
  unaffected.
- `CFF_NN_Fitter` takes `x_pow` as its 4th constructor argument and threads it through every
  `setModel` call site (training via `CustomLoss`, `predict()`, and the three `observ_calc*`
  verification paths), so training and inference apply the identical rescaling.  It is also
  written into `cff_model.json` as `"x_pow"` so the Python CFF-scan notebooks reproduce the
  exact forward.
- **Also in this change:** `weight_decay(1e-3)` was **removed** from the Adam optimizer — the
  regularization was suppressing fit quality on the 16-point dataset.  No replacement
  regularization was added.

`x_pow` is still a manually-set constant per run (`Run_CFF_NN_Fit.cpp` passes `0.0`), not fit
or selected automatically.

---

### Reduced χ²/n loss + Monte Carlo replica ensemble (2026-09-01)

**Loss normalized.**  `CustomLoss` now returns `χ²/n` rather than the raw `Σ((pred−y)/σ)²`
(`normalize = true` by default; `false` is kept only for controlled A/B runs).  The loss is
therefore comparable across dataset and split sizes and reads as a standard reduced-χ²
diagnostic (~1 = good fit).  `predict()`'s reported `chi2` and `cff_model.json`'s
`best_val_chi2` are on the same scale.  Early-stopping `patience` was raised **200 → 1000**,
since the flatter normalized landscape was stopping on noise.

**`fit_once()` — one shared fit routine.**  `train_nn()`'s body (shuffle/split, per-feature
min-max on the training split, fresh net + optimizer, epoch loop with early stopping and
best-val snapshot) was extracted into a private helper that touches no member state and
returns a `TrainedModel`.  The same routine now serves both the central fit and every replica:

```cpp
struct TrainedModel { CFFNNModel net; torch::Tensor X_min, X_max; float best_val_loss; };
struct FitOutcome   { TrainedModel model; bool hopeless; };
FitOutcome fit_once(X, E, phi, y_obs, sigma, bool smear,
        const std::string& learning_curve_path, float hopeless_val_loss,
        int hopeless_check_epoch, unsigned seed, bool normalize_loss = true) const;
```

- `smear = true` fits `y_obs + N(0, σ)` pseudodata (σ itself is never smeared — it stays the
  χ² weight); `smear = false` is the central fit on the raw data.
- **Hopeless-attempt detection:** an attempt is aborted if the validation loss is ever
  non-finite, or is still above `hopeless_val_loss` at an epoch multiple of
  `hopeless_check_epoch` (periodic re-check; `<= 0` disables it, as the central fit does).
- `seed` fixes `torch::manual_seed` (weight init + smear draw) and the split RNG, so an
  identical seed reproduces byte-identical starting conditions.

**`train_replicas(n_replicas, …)`** loops replicas, retrying each `fit_once` attempt up to
`max_tries_per_replica` times and **fully redrawing** on a hopeless attempt (fresh smear +
fresh split + fresh weights, not just a weight reinit).  *(Updated 2026-09-18: 30 tries by
default, and if every try is hopeless the run exports the replicas accepted so far and then
throws — see "Replica retries" below.  Until then, the last hopeless attempt was kept with a
warning so a run always produced exactly `n_replicas` models.)*
`base_seed = 0` (default) draws every attempt's seed from `std::random_device` (production
mode); a nonzero `base_seed` makes the whole ensemble deterministic for A/B comparisons.
Only the **last** replica writes a learning curve (`cff_learning_curve_last_replica.csv`), as
a replica-fit diagnostic — it is fit to smeared pseudodata, so its χ²/n is **not** comparable
to the central fit's against real data.

**`export_replicas(out_dir, name_prefix)`** writes each trained replica as
`<name_prefix><NN>.json` (same format as `cff_model.json`), for out-of-process mean ± σ CFF
bands in Python.  Note the ordering inside a run: the replicas are held in memory during
training and **all** JSONs are written together at the end, after the ensemble finishes —
which is why their timestamps trail the central fit's outputs by the full replica-training
time.

Why smeared pseudodata: a σ from seed-only reruns measures *optimization scatter*, not data
uncertainty (three such runs swung R² from −0.20 to 0.64).  Fluctuating each point per replica
is what makes the band meaningful; the scheme follows Gepard's `datasets_replica_vectloss` /
`train_net_vectloss`.

---

### Batching across data points — implemented (#3, 2026-09-15, branch `vect_optionA`)

The GL swap was the **prerequisite**: DEXP's refinement level is chosen adaptively *per*
kinematic point, so it cannot fit a single static `[N, M]` φ grid, whereas a fixed GL rule
gives the **same M φ nodes for every point** — exactly what batching needs.

The batch dimension was pushed **down into the existing reusable layers** (not into a
monolithic batch entry point), so adding a new observable still means writing only its thin
batched leaf:

- kinematics carried as no-grad `[N]` tensors, **one** `[N,3]` NN forward per batch,
  cross-sections as `[N,M]` (N data points × M φ nodes), the φ-integral reduced over the
  shared grid;
- a batched sibling method on each module (alongside the scalar virtual), plus a
  `computeManyKinematicTorch` service driver mirroring the scalar
  `computeSingleKinematic ↔ computeManyKinematic` pair;
- the **prepare/assemble** split of #1 lifted to `[N]`: public `prepareTensorBatch(xB, t, Q2, E)`
  runs the φ-/helicity-independent `setupKinematicsTorchBatch` once (NN forward + BMJ12
  kinematics + 72 angular coefficients), then the lightweight
  `crossSectionTensorBatch(λ, charge, φ[, VCSSubProcessType])` overloads only sum the selected
  sub-processes.

**Current chain correspondence** (all base-typed pointers + virtual dispatch, as in scalar):

```
ObservableServiceTorch::computeManyKinematicTorch      ↔  ObservableService::computeManyKinematic
   computeSingleKinematicTorch                         ↔     computeSingleKinematic
ObservableTorch::computeTensorBatch (template method)  ↔  Observable::compute
   computeTensorImplBatch (hook)                       ↔     computeObservable
DVCSAluMinusTorch::aLUTensorBatch (pointwise)          ↔  DVCSAluMinus::computeObservable
DVCSProcessModuleTorch::crossSectionTensorBatch (Σ)    ↔  DVCSProcessModule::compute(…,VCSSubProcessType)
   crossSectionBH/VCS/InterfTensorBatch                ↔     CrossSectionBH/VCS/Interf
   setupKinematicsTorchBatch                           ↔     setKinematics + CFF forward
DVCSCFFNNTorch::computeAllCFFsTensorBatch              ↔  DVCSCFFNNTorch::computeCFF
```

`computeTensor` / `computeTensorImpl` (single-kinematic) survive only as thin **N=1 wrappers**
over their `…Batch` siblings — an entry-point convenience, not a separate implementation.

`CustomLoss::forward` no longer loops the rows: it builds a `List<DVCSObservableKinematic>`
once and makes a **single** `computeManyKinematicTorch` call per loss evaluation (one for the
training split, one for validation, per epoch), so the whole batch is one graph with one
`backward()`.  This is **data parallelism** — the multicore speedup comes
for free from ATen's intra-op threading on the batched tensor ops, the path is GPU-ready, and
there are none of the gradient-accumulation races of a hand-threaded per-point loop (option #4,
never implemented and now moot: there is no per-point loop left to thread).

> **Resolved 2026-09-22.**  `DVCSAluMinusTorch::computeTensorImplBatch` is implemented: the
> pointwise A_LU(φ) evaluates each kinematic at its **own** φ, by passing φ as `[N,1]` rather
> than the moment leaves' shared `[M]` grid.  No new machinery was needed — every φ-dependent
> term already broadcasts `[N,1]` kinematics against whatever shape φ has, so the shape alone
> selects the mode.  Fully vectorized, and cheaper than a moment (M = 1 instead of 20).  This
> unblocks the raw-per-φ dataset and the four pointwise A_LU observables.

---

### Dead single-point torch path removed (2026-09-16, commit `82edefd`)

With `computeTensorImpl` reduced to an N=1 wrapper over `computeTensorImplBatch`, nothing drove
the single-point tensor machinery any more.  It was deleted (6 files, +69/−902): `prepareTensor`,
all four non-batch `crossSectionTensor` overloads, the three non-batch sub-process virtuals,
`setupKinematicsTorch` (~410 lines of transcribed BMJ12 kinematics), the non-batch CFF layer and
cached members, and `DVCSAluMinusTorch::aLUTensor`.

This removed a **third full copy of the BMJ12 transcription** — before: base scalar (PARTONS,
doubles, full coverage) + non-batch tensor + batched tensor; after: two.  `m_M` (proton mass) is
the one member both surviving paths share.

**Verification** — full run (central fit + 10 replicas + exports), three paths at
xB=0.2, t=−0.2, Q²=2, E=5.932:

| Path | A_LU^{sin1φ} |
|---|---|
| `observ_calc()` (base PARTONS, scalar) | **0.13186** |
| `observ_calc_torch()` (tensor path) | **0.13186**, `requires_grad = true` |
| `observ_calc_torch_scalar()` (torch scalar virtuals) | **0.13186** |

Note what that check *is*: `observ_calc_torch_scalar()` still routes through torch (the leaf's
`computeObservable` wraps `computeTensor().item()`), so it is a genuine **native-vs-torch
differential test** between two independent implementations of BMJ12 — not a tautology.  Nothing
links the two transcriptions, so a PARTONS upgrade or coefficient fix could silently desync them
and this comparison is the only thing that would notice.  Its limit: it is **one** kinematic
point in the unpolarized sector.

---

### Replica retries and ensemble integrity (2026-09-18)

Two changes to what happens when a replica cannot be fit.

**`max_retries_per_replica` → `max_tries_per_replica`, default 5 → 30.**  The loop always counted
*total* tries, so the old name promised one more attempt than the code gave; 30 means one initial
fit plus up to 29 redraws.

**On exhaustion the run now fails instead of padding the ensemble.**  Previously the last hopeless
attempt was kept with a warning, so a run always produced exactly `n_replicas` models — one of
which could be junk, silently widening the uncertainty band.  Now `train_replicas` exports the
replicas accepted so far (that compute is worth keeping) and then throws a `std::runtime_error`
naming the replica, the try count and how many were exported.

**`export_replicas` deletes the previous `<prefix>*.json` before writing** — unless there is
nothing to write, in which case the earlier ensemble is left alone.  Without this, a 10-replica run
followed by a 5-replica run left `_05`…`_09` on disk and a `glob` in the plotting notebook read ten
replicas, five of them from a different fit.

For reference, Gepard's `fitter_vectloss.py` takes neither route: `fit()` retries with **no cap**
(`while test_err < 0`), so exhaustion cannot occur; `fitgood()` caps tries **globally** across the
ensemble and, when the budget runs out, simply `break`s and returns fewer nets with no error.  Both
discard the failed net and keep the successes, as here — only the ending differs.

---

### A tensor interface for the CFF link, and a scalar-model adapter (2026-09-21)

The torch chain's bottom link was pinned to a single implementation:
`setupKinematicsTorchBatch` cross-cast its convol-coeff module to the **concrete**
`DVCSCFFNNTorch`, because — unlike the observable and process links — the CFF link had no
torch base to cast to.  The 2026-06-16 rework introduced `ObservableTorch<K>` and
`ProcessModuleTorch<K>` but never the CFF twin, since there was only ever one implementation.

**`DVCSCFFModuleTorch`** fills that gap: a pure mixin owning `AllCFFsTensorBatch` and one pure
virtual `computeAllCFFsTensorBatch(xi, t, Q2, muF2, muR2)`, sitting under a generic
`CFFModuleTorch<K>` so that every link now has a generic template with a channel class beneath
it.  `DVCSCFFNNTorch` derives from it alongside the PARTONS module, so every link pairs a PARTONS
class (identity, registration, the scalar contract) with a torch base (the tensor interface).

The signature carries the **CCF kinematics**, the five quantities
`DVCSConvolCoeffFunctionKinematic` holds, because that is what the scalar chain hands its CFF
module: `DVCSProcessModule::computeConvolCoeffFunction` runs the xi-converter and the scales
module first.  The process converts, the CFF module receives — and the torch chain mirrors that,
so a source parameterized in xB (the network) converts back itself with one tensor op.

**`DVCSCFFScalarTorch`** is the first second implementation: it presents any scalar PARTONS CFF
model as a tensor CFF source, evaluating it per point and returning `[N]` **no-grad** complex
tensors.  Nothing downstream minds — the chain multiplies CFF tensors by no-grad kinematics
either way, and the observable simply comes back detached.  It receives CCF kinematics already
converted, so it has only to build the bean and call the model.  It carries the same dual base as
`DVCSCFFNNTorch` (PARTONS module + torch mixin), so it is attached with
`setConvolCoeffFunctionModule()` like any other CFF module rather than through a wiring path of
its own.

**Since 2026-10-02 the adapter is optional.**  `DVCSProcessBMJ12Torch` accepts any PARTONS CFF
module directly, exactly as the scalar process does: when the attached module does not implement
`DVCSCFFModuleTorch`, the process evaluates it through the adapter's static
`evaluateScalarBatch()` — the same per-point loop, one implementation — and packs the results
into no-grad tensors.  Nothing is lost, since a parametric model has no parameters in the graph.
Verified by running all 28 differential tests both ways: the outputs are identical line for line.

**What it buys** is `observ_calc_scalar_cff()`: fixed CFFs (`DVCSCFFConstant`) pushed through
PARTONS' native process module *and* through `DVCSProcessBMJ12Torch`, so the two sides share
nothing but four constant numbers.  Until now the native-vs-torch check ran the same trained
network on both sides, which cannot isolate the process layer.  It scans **every point of the
dataset** and drives the torch side through `computeManyKinematicTorch`, making it also the only
check that exercises the batched `[N,M]` path at N>1.

Across the 16-point CLAS07 file, most points agree to ~10⁻⁵ relative and the worst reaches
4.2×10⁻⁴.  That residual was **measured**, not assumed, to be φ-quadrature error, by raising the
torch integrator order and re-running the scan:

| torch φ-integrator | max relative deviation |
|---|---|
| GL-10 (the default at the time) | 4.2×10⁻⁴ |
| GL-20 | 1.2×10⁻⁸ |
| GL-40 | 1.8×10⁻¹³ |
| GL-80 | 1.7×10⁻¹³ (double-precision floor) |

So the two independent BMJ12 transcriptions agree to ~2×10⁻¹³ once φ is resolved — the strongest
validation the torch port has had.  It also corrects the 2026-06-22 claim that GL-10 reproduces
DEXP to ~6 significant figures: that was one kinematic point, and across the dataset it is ~3.4.
Still far below the data's own 6% precision, so no fit result is affected — but the margin is
100× smaller than advertised.

**The default was therefore raised to GL-20** (2026-09-21), moving the worst-case agreement to
~1.2×10⁻⁸ for twice the φ nodes — and **again to GL-40 on 2026-09-22**, once the sin(2φ) moments
showed that an order chosen for sin(1φ) is not uniformly safe for the family (see below).
The extra nodes turn out to be **free**: normalizing two full pipeline runs by their logged
epochs gives 31.640 ms per epoch-line at GL-10 against 31.660 ms at GL-20, a difference of
**+0.06%** — inside the noise.  That confirms the scaling argument above from the other side: what
costs time is the number of tensor operations, not how many elements they hold, and M enters the
`[N,M]` tensors exactly as N does.

A side effect worth knowing: the three `observ_calc*` paths now agree to **every printed digit**
(0.133156 / 0.133156 / 0.133156).  The 6th-significant-digit spread that these notes have
attributed to "the GL-vs-DEXP gap" since 2026-06-22 was never a floor — it was GL-10's quadrature
error, and it vanishes once φ is resolved.

Two things `DVCSCFFConstant` taught us, both now in comments: the native BMJ12 process requests
**every** GPD type the module advertises — transversity, twist-3, even the DDVCS `HL` — and
`setCFFs()` *replaces* the map rather than merging, so handing it four entries makes it throw on
the fifth type requested.  Start from the module's own pre-zeroed map and overwrite the four
twist-2 entries; those zeros are also exactly what the torch port assumes.

---

### The A_LU family, complete (2026-09-22)

All nine PARTONS A_LU observables now have torch twins, each verified against the class it
mirrors.  Four **pointwise** variants — `AluMinus`, `AluPlus`, `AluDVCS`, `AluInt` — plus five
**Fourier moments** on top of them.

The four differ by **charge combination**, not by sub-process selector, which is the thing to
know before adding more.  Writing σ(λ, charge):

| Observable | Formula |
|---|---|
| `AluMinus` | (σ₊₋ − σ₋₋) / (σ₊₋ + σ₋₋) |
| `AluPlus` | (σ₊₊ − σ₋₊) / (σ₊₊ + σ₋₊) |
| `AluDVCS` | ((σ₊₊+σ₊₋) − (σ₋₊+σ₋₋)) / ((σ₊₊+σ₊₋) + (σ₋₊+σ₋₋)) |
| `AluInt` | ((σ₊₊−σ₊₋) − (σ₋₊−σ₋₋)) / ((σ₊₊+σ₊₋) + (σ₋₊+σ₋₋)) |

The charge **sum** cancels the interference term (odd in beam charge), leaving BH+VCS — hence the
"DVCS" label; the charge **difference** isolates it.  So `aLUTensorBatch` prepares the process
module once and delegates to an `asymmetryTensorBatch()` hook that assembles only the cross
sections its own formula needs, mirroring the scalar classes.

Each is a **sibling** of `DVCSAluMinusTorch` rather than a subclass: every variant must *be* its
own PARTONS observable for the scalar chain, exactly as PARTONS' own classes are siblings.

**Verification** — all nine against native PARTONS with identical fixed CFFs, over the 16-point
dataset:

| Observable | Kind | max relative |
|---|---|---|
| `AluMinusSin1Phi` | moment | 1.2×10⁻⁸ |
| `AluMinus`, `AluPlus`, `AluInt` | pointwise | ~3×10⁻¹⁵ |
| `AluDVCS` | pointwise | **0** — vanishes identically |
| `AluMinusSin2Phi` | moment | 4.8×10⁻⁷ |
| `AluDVCSSin1Phi` | moment | vanishes identically |
| `AluIntSin1Phi` | moment | 1.2×10⁻⁹ |
| `AluIntSin2Phi` | moment | 1.1×10⁻⁷ |

Two things that came out of it.  The **sin(2φ) moments are ~40× looser** than sin(1φ) under the
same GL-20 rule — higher harmonic, same node count — which is why the integrator order should be
re-validated per integrand rather than assumed.  **The order was raised to GL-40 in response**:
sin(1φ) improves 1.2×10⁻⁸ → 1.8×10⁻¹³ and sin(2φ) 4.8×10⁻⁷ → 7.0×10⁻¹², while GL-80 buys nothing
further and can be *worse* past the numerical floor.  The point is not the physics — GL-20 was
already five orders below the data's precision — but the **test**: at GL-20 a transcription bug
below ~5×10⁻⁷ would hide inside the quadrature residual, while at GL-40 the detection threshold is
~10⁻¹¹, which matters with 50 observables still to port.  Measured cost: **+1.26%** per epoch.  And **two observables vanish identically** with
these CFFs, for the physical reason above; the test's relative metric was dividing noise by noise
and reporting a spurious failure, so it now takes that statistic only where |native| > 10⁻¹² and
says so explicitly otherwise.  That matters for the remaining 50 observables, many of which will
vanish in some configuration.

### The A_C family (2026-09-22)

Five more leaves — `DVCSAcTorch` plus `DVCSAcCos0/1/2/3PhiTorch`.  A_C is the **transpose of
A_LU^DVCS**: it sums over beam *helicity* at each charge, then differences the *charge*, where
AluDVCS sums over charge and differences helicity.  Writing σ(λ, charge):

> A_C = ((σ₊₊+σ₋₊) − (σ₊₋+σ₋₋)) / ((σ₊₊+σ₋₊) + (σ₊₋+σ₋₋))

Since the interference term is odd in beam charge while BH and VCS are even, this collapses to
Ī / (BH̄ + VCS̄) — the interference isolated against the BH+DVCS background.  That is why its
moments are **cosine** moments (Re CFFs) where A_LU's are sine moments (Im CFFs).

**The moments looked broken and were not**, and the way that was settled is the reusable part.
Against native PARTONS: pointwise 4.6×10⁻¹⁵, but `AcCos0Phi` 9.9×10⁻⁸, `AcCos1Phi` 4.6×10⁻⁸,
`AcCos2Phi` 1.4×10⁻⁶ — five orders looser than the A_LU moments at the same GL-40.  Three checks,
in increasing order of strength:

1. **Raise our own order.**  Flat across GL-40/80/160, so our quadrature had converged.  Necessary
   but not sufficient — agreement within one rule family can hide a bias the whole family shares.
2. **Re-run under a different rule *family*.**  `TRAPEZOIDAL-64` reproduced GL-40 to every printed
   digit on both outliers.  Two completely different node distributions agreeing is what rules
   out a shared bias.
3. **Take our code out of the loop entirely.**  Integrate PARTONS' *own* pointwise `DVCSAc` over φ
   with GL-200 and compare against PARTONS' *own* DEXP moment classes.  It reproduced every
   residual to six digits (1.43964195×10⁻⁶ vs 1.439642×10⁻⁶).

So the residual is the scalar side's.  Two reasons it is the weaker of the two here: PARTONS never
calls `setTolerances()`, so DEXP's absolute tolerance is its default **0.0**, its convergence test
can never be satisfied, and every call runs to the end of its node table logging
`"Cannot reach tolerances !"`.  And DEXP is **tanh-sinh, built for endpoint singularities** — it
clusters nodes double-exponentially at the ends and samples the interior sparsely, the wrong shape
for a smooth 2π-periodic asymmetry.  Not a rule that stopped early; the wrong rule for the job.

**`spread_phi`** came out of this.  Every row of the data file carries the *same* φ, so a pointwise
leaf scanned over the dataset was tested at exactly one angle — a charge combination wrong
elsewhere in φ would have passed.  `observ_calc_scalar_cff(..., spread_phi = true)` sweeps φ over
[0, 2π) instead; swept, `DVCSAc` holds at 1.5×10⁻¹⁴.

---

### The cross-section family (2026-09-23) — the unpolarized sector complete

Eight leaves: five pointwise (`UUMinus`, `DifferenceLUMinus`, `UUBHSubProc`, `UUDVCSSubProc`,
`UUVirtualPhotoProduction`) and three `PhiIntegrated` on top of their parents.  All five share one
skeleton, differing only in a sign and a `VCSSubProcessType`:

> ½[σ(λ=+1) ± σ(λ=−1)] · 2π · C   at beam charge −1, unpolarized target

The `/2` is a genuine **average** (unpolarized beam) where an asymmetry divides by the *sum*; the
2π integrates out the transversely-polarized-target azimuth.  `DifferenceLUMinus` is the odd one
out — helicity-**odd**, so not an unpolarized cross section despite the family name.

**These are the first dimensionful observables in the chain.**  The process module works in GeV⁻²
and every PARTONS cross-section class converts with `makeSameUnitAs(PhysicalUnit::NB)`; the torch
chain carries no unit system, so the conversion is explicit (`Constant::CONV_GEVm2_TO_NBARN`) and
the scalar wrapper tags its result `NB`.  `DVCSCrossSectionTotal` is deliberately **not** ported —
a GSL VEGAS Monte Carlo over (y, Q², t) calling back into the scalar observable, not a
tensor-chain shape.

Verification: all five pointwise at 10⁻¹⁶–3×10⁻¹⁵ (both at the data φ and swept).  The
φ-integrated ones needed real work, and two lessons came out of it.

**A flat max-over-dataset residual does not mean your side has converged.**
`UUMinusPhiIntegrated`'s max sat at 2.4848×10⁻⁴ from GL-40 all the way to GL-640 — by the usual
rule, "converged".  True of the *maximum*: it was pinned by one point where the scalar side is the
outlier, while another point was still converging underneath.

| point | GL-40 | GL-80 | GL-160 | GL-320 |
|---|---|---|---|---|
| xB=0.25, t=−0.488 | **3.0×10⁻⁵** | 3.0×10⁻⁹ | 2.8×10⁻¹¹ | 4.1×10⁻¹² |
| two others | 8.1×10⁻⁶ / 2.5×10⁻⁴ | identical | identical | identical |

**Order is per-leaf, even inside one family.**  Before blaming quadrature the φ→0 corner got its
own check — a log-spaced sweep to φ = 10⁻⁸, inside the BH peak, showed the two implementations
agree to 2×10⁻¹⁶ there.  A direct φ profile then confirmed the peak is real and is BH: ~5900× the
value at φ=π, 99.3% Bethe-Heitler, while the DVCS sub-process varies only ~35% across the whole
range.  So `UUMinusPhiIntegrated` gets **GL-160**, and its two sub-process siblings stay at
**GL-40**, where they are not merely adequate but *optimal* — raising them degrades
3.7×10⁻¹⁵ → 8.6×10⁻¹³.

---

### Two measurements that corrected earlier claims (2026-09-22/23)

**The prepare/assemble split is worth more than recorded, for a different reason.**  The 2026-06-24
work was justified by counting operations, never timed, and assumed the assemble was "lightweight".
Measured at N=16, M=40 (3 runs × 300 reps): prepare 2.49–2.68 ms, assemble 2.27–2.56 ms — **the
same**, ratio 1.05–1.12.  Parity is exactly *why* the split pays: every avoided re-preparation
costs as much as the call that remains.  Dropping it would cost **+34–36%** per A_LU evaluation
and **+62–65%** per A_C, which has four cross sections per prepare.  (Process-layer figures; an
epoch also pays the integrand, χ² and `backward()`.)

**GL-20 and GL-40 are privileged orders in NumA.**  Chasing "why does a higher order make agreement
*worse*" turned up a library defect that governs every future order choice.
`GaussLegendreIntegrator1D` hardcodes 16-digit tables for N = 20 and N = 40 **only**; everything
else uses its Newton solver, whose **weights are ~100× worse** (N=40 tabulated: max |Δw|
1.25×10⁻¹⁵, Σw−2 exactly 0; N=80/160/320 computed: ~10⁻¹³ and ~10⁻¹²).  The nodes are fine either
way.  The cause is a defect, not a precision limit: the solver stores `2/((1−z²)·pp·pp)` pairing
the final node with `pp = P'_N` evaluated one Newton step earlier, up to `EPS = 1e-12` away, and
`w ~ 1/P'_N²` amplifies that by `2(P″/P′) = 4z/(1−z²)` = O(N²) at the outermost nodes.  Verified
both ways — re-evaluating `P'_N` at the converged node, or tightening EPS to 1e-15, each recovers
the full ~100×.

So **leaving 20 or 40 is a step change in rule quality, not gradual accumulation**, and a higher
order can agree with the scalar path worse than GL-40 did.  This retro-explains several residuals
previously written off as "the floor", including `AluIntSin2Phi` going 6.5×10⁻¹² → 1.9×10⁻¹⁰ at
GL-80.  Full write-up on `setIntegrator()` in `MathIntegratorModuleTorch.h`.  Fixable on our side
if it matters — we only *read* NumA's nodes and weights — but it sharpens the test without
changing any physics, so it is not done.


### Units carried through the tensor chain (2026-09-23)

The torch chain returned bare tensors while the scalar chain carries `PhysicalType<double>`
internally — `DVCSProcessModule::compute` accumulates into a
`PhysicalType<double> value(0., PhysicalUnit::GEVm2)`.  That was the **last place the two chains
differed in shape**, and it began to matter with the cross sections: the first dimensionful
observables, whose GeV⁻² → nb step nothing checked.

`PhysicalType` is a plain template with no constraint on its value type, and its members
instantiate lazily, so `PhysicalType<torch::Tensor>` needs no patch to PARTONS.  That was an
expectation rather than a result, so it was **verified before anything was built on it**:
`operator+` and `makeSameUnitAs` preserve `requires_grad`, `backward()` reaches the leaf with the
exact chain-rule factor (12·C = 4672551.6), and `checkIfSameUnitAs` fires on a mismatch.  A
`torch::Tensor` is a refcounted handle, so `PhysicalType`'s by-value storage shares the graph
rather than copying it.

**What it looks like now:**

| Layer | Returns |
|---|---|
| `computeSingleKinematicTorch` / `computeManyKinematicTorch` | `ObservableResultTorch<K>` |
| `computeTensor` / `computeTensorBatch` | `ObservableResultTorch<K>` |
| `computeTensorImpl` / `computeTensorImplBatch` | `PhysicalType<torch::Tensor>` |
| `crossSectionBH/VCS/InterfTensorBatch` | `PhysicalType<torch::Tensor>` (GeV⁻²) |

`ObservableResultTorch<K>` (alias `DVCSObservableResultTorch`) is a **sibling** of PARTONS'
`ObservableResult`, not an instantiation: that class's payload is a hardcoded
`PhysicalType<double>` with no template parameter for the value type, and its base `Result<K>`
holds a *singular* kinematic and needs `operator<` and a `const toString()` that `List<K>` lacks.
One bean per **batch**, where the scalar `computeManyKinematic` returns one per point.

**Two payoffs beyond the check itself.**  Asymmetries get their unit **derived rather than
asserted** — `PhysicalType::operator/` tags every quotient `NONE`, so `(σ⁺−σ⁻)/(σ⁺+σ⁻)` is
dimensionless *because it is a ratio*.  And the cross sections lost their hand-copied constant:

```cpp
return sigma.makeSameUnitAs(PARTONS::PhysicalUnit::NB);   // was: * CONV_GEVm2_TO_NBARN
```

so the conversion is declared, not transcribed, and cannot drift from PARTONS'.

**Verification.**  Three-path agreement to every digit (0.128611), tensor path keeps
`requires_grad = true`, and the 22-observable differential test is unchanged at **17 of 22**.  The
five that moved are exactly the cross sections, all at ~10⁻¹⁶ and all *improved*
(`UUMinus` 4.58×10⁻¹⁶ → 3.05×10⁻¹⁶) — because `makeSameUnitAs` divides by 1/C where the old code
multiplied by C, so our conversion is now bit-for-bit the one the scalar side performs.

**One trap, hit once.**  `getTensor()` first returned `const torch::Tensor&`, but
`PhysicalType::getValue()` returns `T` **by value** — the reference bound to a temporary and
dangled.  With a refcounted handle that aborts with `pointer being freed was not allocated`
rather than merely reading garbage.  Returns by value now, which costs nothing.

51 files changed and **no `CMakeLists.txt` edit** — every new file is a header-only template.


### Comparison notebooks, one per dataset (2026-09-24)

`My_Analysis/Codes/` holds three, structurally identical — learning curves → predicted-vs-measured
→ residuals → pulls → CFF scans with the replica band → Gepard overlay:

| Notebook | Partons fit | Gepard replicas |
|---|---|---|
| `CFF_plots_ALU_2007_xpow_replica.ipynb` | $A_{LU}^{\sin 1\phi}$, ImH | `gepard_imh_replica_*.json` |
| `CFF_plots_AC_2012_xpow_replica.ipynb` | $A_C^{\cos 0\phi}$, ReH | `gepard_reh_replica_*.json` |
| `CFF_plots_XLU_2018_xpow_replica.ipynb` | XLU, ImH | `gepard_imh_replica_*.json` |

The two new ones derive their scan ranges **from the data file** instead of hardcoding CLAS
numbers, and fix the scan slices at the dataset's mean kinematics — the same convention Gepard's
own `CFF_plots_*_xpow.ipynb` uses, so both sides are compared on identical slices.  Figures carry
`_AC` / `_XLU` suffixes so the three sets do not overwrite each other.

**⚠️ The A_LU notebook's Gepard panels are suspect.**  `gepard_imh_replica_*.json` carries
`dataset: {id: 163, observable: XLU, CLAS, 2018}` — those are the **XLU** replicas.  Correct for
the XLU notebook; but the A_LU notebook loads the same files while its markdown claims *"the same
CLAS 2007 ALU dataset"*.  Most likely the XLU fit regenerated them under the same name.  The
`_BMK`, `_moredat` and `_BMK_moredat` variants have no `dataset` field, so one of those may be the
original A_LU set.  **Unresolved.**

**A plotting trap.**  `predicted vs measured` renders as an invisible hairline on the CLAS 2018
set: `errorbar(..., xerr=error)` autoscales x to include the error bars, three points have `error`
up to 1815 on values spanning [−1.1, 1.8], giving `xlim = ±1996` against `ylim = ±1`, and
`set_aspect('equal', 'box')` then squashes the axes to 2000:1.  Fixed in the XLU notebook by
setting limits from the *values*.  The other two are unaffected (errors comparable to values).


---

## Current status / open items

- ~~Raw per-φ A_LU leaf~~ **resolved 2026-09-22** — `DVCSAluMinusTorch::computeTensorImplBatch`
  is implemented: φ passed as `[N,1]` instead of the moment leaves' shared `[M]` grid, which the
  assembly already broadcasts.  The raw-per-φ dataset has nothing blocking it.
- **The unpolarized-target sector is complete** — 22 of PARTONS' 59 DVCS observables: A_LU (9),
  A_C (5), cross sections (8).  `DVCSCrossSectionTotal` is deliberately skipped (GSL VEGAS Monte
  Carlo, not a tensor-chain shape).  The remaining **36 polarized-target observables** all need
  the LP/TP coefficient rows in `setupKinematicsTorchBatch` and should be a separate issue —
  that is now the single blocker for the rest of the port.
- **`observ_calc_scalar_cff` reports only the max over the dataset**, which is what hid the
  cross-section under-resolution (flat at 2.4848×10⁻⁴ from GL-40 to GL-640 while a point
  underneath was still converging).  A per-point summary — worst *n*, or a flag on any point above
  a threshold — would stop that recurring across the remaining 37 observables.
- **Unpolarized-target only** on the tensor path — the torch BMJ12 port omits the LP/TP
  coefficient rows.  Correct for A_LU and siblings; for polarized-target observables use the
  base PARTONS classes.  The determining factor is the **observable leaf**, not the process
  module (a `*Torch` leaf routes into the tensor physics however you drive it).  See
  `CLAUDE.md` for the full caveat table.
- ~~Torch chain carries no unit system~~ **resolved 2026-09-23** — the chain now carries
  `PhysicalType<torch::Tensor>` and returns an `ObservableResultTorch<K>` bean, so the GeV⁻² → nb
  conversion is a `makeSameUnitAs()` call rather than a hand-copied constant and a unit mismatch
  throws.  The bean still omits `Result<K>`'s channel-type and result-info fields, deliberately:
  they serve PARTONS' database and report serialization and would be write-only here.
- **BMJ12 differential tests are OFF in the runner** — `const bool runBMJ12DifferentialTests =
  false` in `Run_CFF_NN_Fit.cpp`.  The calls are kept, not deleted.  **Turn them back on after any
  change to `DVCSProcessBMJ12Torch`, to a leaf's formula, or when adding an observable**, and point
  the fitter at a small data file while doing so.  They are the only thing linking the two
  independent BMJ12 transcriptions, so they are a regression test, not a one-time validation.
  They were disabled because the *native* side loops per data point — ~4 s on 16 points, but
  28 × 3008 ≈ 84 000 scalar evaluations on the CLAS 2018 set.  A `--selftest` flag with its own
  fixed small file would decouple the test from the fit's dataset; proposed and not taken.
- **The 3 → 6 → 1 network underfits the 3008-point set** (χ²/n 1.68, train ≈ val, flat from epoch
  ~1000).  Try adding `ImE`/`ImHt` to the output layer first — XLU's interference term involves
  all three — then more hidden neurons, then a non-zero `x_pow`.
- **`x_pow` is a manual constant** — not fit or selected automatically, and no systematic
  comparison of values has been recorded.
- **Replica hyperparameters untuned** — `hopeless_val_loss = 100`, `hopeless_check_epoch = 200`,
  `max_tries_per_replica = 30` are initial defaults.  Note the threshold is only *checked* at
  epoch multiples of 200, by which point a healthy fit sits near χ²/n ≈ 5 — so 100 catches a
  stuck or diverged fit, not a merely poor one.  An observed 10-replica run showed
  pronounced overfitting well before early stopping fired (train χ²/n → 0.62 while val climbed
  to 8.28, best val around epoch ~620 with `patience = 1000` then running ~1000 epochs uphill),
  so the replica band is likely wider than the data alone justifies.  Untested hypothesis,
  flagged for whoever tunes this next.
- **Replicas are trained sequentially** — `n_replicas × (1 + retries)` full fits.  Easier to
  parallelize than the old per-point idea (each replica owns its net/optimizer/graph, so there
  is no shared-gradient race), but not done.
- **Absolute paths are hardcoded** — `CFF_NN_Fitter::OUT_DIR` and the data path in
  `Run_CFF_NN_Fit.cpp`.  Both must be updated on an environment move.
- **`predict()` still loops per point** (`computeSingleKinematicTorch`) while training uses the
  batched driver.  Harmless — it runs once per fit, not per epoch.
- **A failed run still exits 0** — `main()` catches, logs through PARTONS' logger and falls
  through to `return 0`, so SWIF/Slurm marks the job succeeded and `./bin/Run_CFF_NN_Fit && …`
  continues onto an incomplete ensemble.  The `.out` file does end with the
  `[ERROR] (main::main) Replica N still hopeless …` line, so a human reading the log sees it.
  Fix: an `int exit_code` set in both catch blocks and returned at the end.
- ~~φ-quadrature order~~ **resolved, with caveats**: 10 → 20 (2026-09-21, +0.06%/epoch) → 40
  (2026-09-22, +1.26%/epoch) for the asymmetry leaves.  But the order is **per-leaf**, not global
  — `DVCSCrossSectionUUMinusPhiIntegratedTorch` needs GL-160 — and **20 and 40 are privileged
  orders in NumA** (see above), so any new choice must be re-measured rather than reasoned about.

---

## Build

```bash
cd build && cmake .. && make -j$(nproc)
```

Single target:
```bash
cd build && make Run_CFF_NN_Fit
```

Executables are placed in `bin/`.  Run them **from the project root**, not from `bin/`:

```bash
./bin/Run_CFF_NN_Fit
```

`Partons::init` derives the properties-file directory from `argv[0]`, but the paths *inside*
`bin/partons.properties` (`log.file.path = bin/logger.properties`,
`xml.schema.file.path = data/xmlSchema.xsd`) are resolved against the **actual working
directory** — so `cd bin && ./Run_CFF_NN_Fit` makes them resolve to `bin/bin/…` and fails on
`logger.properties`.

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

The directory is hardcoded as `CFF_NN_Fitter::OUT_DIR` (an absolute path — update it, and the
data path passed to the constructor, on an environment move).  Every file is opened with
`std::ios::trunc`, so each run overwrites the previous one's outputs.

| File | Content | Written by |
|---|---|---|
| `cff_learning_curve.csv` | epoch, train reduced χ²/n, val reduced χ²/n (every 2 epochs) — **central fit** | `train_nn()` (via `fit_once()`) |
| `obs_prediction.csv` | `xB,t,Q2,E,phi,obs_true,obs_pred,error` per point | `predict()` |
| `obs_model_eval.csv` | `observable,mse,r_squared,chi2` (chi2 = reduced χ²/n) | `predict()` |
| `cff_model.json` | trained NN export (`arch`, `dtype`, `best_val_chi2`, `input_features`, `x_pow`, `output_layer`, min-max `scaling`, `fc1`/`fc2` weights+biases) — reproduces the exact forward in Python | `predict()` |
| `cff_learning_curve_last_replica.csv` | same format, for the **last replica only** — a replica diagnostic, fit to smeared pseudodata, so **not** comparable to the central fit's χ²/n | `train_replicas()` |
| `cff_model_replica_<NN>.json` | one per trained replica, same format as `cff_model.json` — for Python mean ± σ CFF bands | `export_replicas()` |
| `dvcs_DVCSAluSinPhi_BSACLAS15_ANN.csv` | Mean ± σ observable per kinematic point (replica ensemble, from `ObsCalc_CFFNNReplicas`) | `ObsCalc_CFFNNReplicas` |
| `dvcs_DVCSAluSinPhi_ANN_replicas.csv` | Individual replica values for every kinematic point (from `ObsCalc_CFFNNReplicas`) | `ObsCalc_CFFNNReplicas` |

---

## Data format

Input files are pipe-separated (`|`).

**Observable format** (what `CFF_NN_Fitter` fits today):

```
xB | t | Q2 | E | phi | <observable> | error
```

`load_data_observable()` returns an `ObservableData` struct: `X[N,3] = (xB, t, Q²)`, `E[N]`,
`phi[N]`, `y_obs[N]` (field 6), `sigma[N]` (field 7) and **`observableName`** — the header's
6th field.  `error` **is** used: it is the σ in the reduced-χ²/n loss.

### ⚠️ The header selects the observable (2026-09-24)

The 6th field must be the **PARTONS scalar observable class name** — `DVCSAluMinusSin1Phi`,
`DVCSAcCos0Phi`, `DVCSCrossSectionDifferenceLUMinus`, … — not a free-form label.  The tensor
leaf wired for the fit is that name **+ `"Torch"`**, resolved through
`ModuleObjectFactory::newDVCSObservable(const std::string&)`, so there is no mapping table.

The scalar name rather than the torch one because the file then describes *physics* rather than
our implementation, and because it yields **both** classIds — the native one for
`observ_calc_scalar_cff()` and the tensor twin by suffix.

Before this the header was read and **thrown away** while `CustomLoss` hardcoded
`DVCSAluMinusSin1PhiTorch`, so pointing the fitter at a file of A_C data silently fitted A_LU to
it and reported nothing worse than a poor χ².  `CustomLossImpl`'s `observableName` is therefore
**required, not defaulted**.  An unresolvable name throws at startup naming the file and both
spellings — so the 37 observables with no torch twin now fail loudly rather than silently running
the unpolarized port.

The header is **validated**: exactly 7 fields, the 7th named `error`.  That closes a latent bug by
construction — σ is read with `rows[i].back()`, so on a 6-column file it would silently have been
the observable itself.  Only `*_error.csv` files are fittable.

φ is loaded and passed into the kinematics.  A **moment** leaf integrates it away; a **pointwise**
leaf evaluates each row at its own φ, so for those the φ column is live data.

### Datasets fitted

Switching between them is two lines in `Run_CFF_NN_Fit.cpp` (path + output layer); the observable
follows the header.

| File | Observable | N | Output | Result |
|---|---|---|---|---|
| `BSA_CLAS_07_…_ALU_error.csv` | `DVCSAluMinusSin1Phi` | 16 | ImH | R² 0.79, χ²/n 0.28 |
| `BCA_HERMES_12_…_AC_cos0phi_error.csv` | `DVCSAcCos0Phi` | 18 | **ReH** | R² 0.89, χ²/n 0.25 |
| `BSD_CLAS_18_…_XLU_phi_error.csv` | `DVCSCrossSectionDifferenceLUMinus` | **3008** | ImH | R² 0.76, χ²/n 1.68 |

**ReH for A_C** because its *cosine* moments carry the **real** parts of the CFFs; **ImH** for the
two helicity-odd observables, whose *sine* harmonics carry the imaginary parts.

The CLAS 2018 set is the first **per-φ** dataset (230 distinct φ over 1250 kinematic bins) and the
first large one.  Two things it showed: the batched chain scales well — **188× the points for 3.8×
the time per epoch** (61 ms vs ~16 ms), because a pointwise leaf runs at M = 1 where a GL-40 moment
runs at M = 40 — and the 3 → 6 → 1 network **underfits** it (train ≈ val to three digits, flat from
epoch ~1000) where the same net *over*fits a 16-point file.  Suspects in order: only ImH is fitted
while XLU's interference term involves ImH, ImE and ImH̃; 6 hidden neurons; `x_pow = 0` across
xB 0.124–0.500.

The older **CFF-label format** (`xB | t | Q2 | … | ImH | ReH | … | error`, labels matched by
column name from `output_layer`) is no longer read by `CFF_NN_Fitter` — its loader was removed
in 2026-06-17 when the workflow switched to fitting the observable.  `NN_Fitter::load_data()`
in `NN_Fit.{h,cpp}` still uses it for the separate `NN_CFF_fit` executable.