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
`ProcessModule<K,R>` / `ObservableService<K,R>`; `ResultType` collapses to `torch::Tensor`,
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

> **Gap:** `DVCSAluMinusTorch::computeTensorImplBatch` is still a **throwing placeholder**, so
> a bare `DVCSAluMinusTorch` cannot be used (its scalar `computeObservable` throws too).  Only
> the moment leaf `DVCSAluMinusSin1PhiTorch` is wired.  A real pointwise implementation needs
> each of the N kinematics paired with its **own** φ (an `[N]` broadcast), whereas the existing
> machinery broadcasts φ as an `[M]` axis *shared* across all N points (the `[N,M]` outer
> product, correct for quadrature over the sin1φ moment but not for raw per-φ data).  This is
> the work item for the planned raw-per-φ A_LU dataset.

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
~1.2×10⁻⁸ for twice the φ nodes.  Not GL-40: at GL-20 the quadrature residual already sits ~5
orders of magnitude below the data's own 6% precision, so further nodes buy nothing observable.
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

## Current status / open items

- **Raw per-φ A_LU leaf** — `DVCSAluMinusTorch::computeTensorImplBatch` is a throwing
  placeholder, so a bare `DVCSAluMinusTorch` is unusable (both its tensor and its inherited
  scalar entry points throw).  It needs own-φ `[N]` semantics rather than the shared-`[M]`
  quadrature broadcast.  Blocks the planned dataset that fits raw per-φ A_LU instead of the
  sin1φ moment.
- **Unpolarized-target only** on the tensor path — the torch BMJ12 port omits the LP/TP
  coefficient rows.  Correct for A_LU and siblings; for polarized-target observables use the
  base PARTONS classes.  The determining factor is the **observable leaf**, not the process
  module (a `*Torch` leaf routes into the tensor physics however you drive it).  See
  `CLAUDE.md` for the full caveat table.
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
- ~~φ-quadrature order~~ **resolved 2026-09-21**: raised 10 → 20, taking the worst-case deviation
  from adaptive DEXP from 4.2×10⁻⁴ to ~1.2×10⁻⁸ at a measured cost of +0.06% per epoch.

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

`load_data_observable()` returns `(X[N,3] = (xB, t, Q²), E[N], phi[N], y_obs[N] = col 5,
sigma[N] = last col)`.  Training and prediction operate on the **observable**
(A_LU^{sin1φ}), and `error` **is** used — it is the σ in the reduced-χ²/n loss.  φ is loaded
and passed into the kinematics; the sin1φ moment integrates it out, but it is kept so
`CustomLoss` is reusable for φ-dependent observables.  Current file:
`Data/Partons_input/BSA_CLAS_07_KK_format_ALU_error.csv` (16 points).

The older **CFF-label format** (`xB | t | Q2 | … | ImH | ReH | … | error`, labels matched by
column name from `output_layer`) is no longer read by `CFF_NN_Fitter` — its loader was removed
in 2026-06-17 when the workflow switched to fitting the observable.  `NN_Fitter::load_data()`
in `NN_Fit.{h,cpp}` still uses it for the separate `NN_CFF_fit` executable.