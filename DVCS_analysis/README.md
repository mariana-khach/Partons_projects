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
│       └── Theory/         # Differentiable physics layer (libtorch + PARTONS subclasses)
│           ├── Beans/Obs/DVCS/         # DVCSKinematicsTorch.cpp
│           └── Modules/                # PARTONS-registered tensor modules
│               ├── CFFs/DVCS/          # DVCSCFFNNTorch.cpp
│               ├── Processes/DVCS/     # DVCSAmplitudesBMJ12Torch.cpp,
│               │                       # DVCSProcessBMJ12Torch.cpp
│               └── Obs/DVCS/           # DVCSAluMinusSin1PhiTorch.cpp
├── include/                # Headers mirroring src/
├── bin/                    # Compiled executables (CMake output)
├── My_Analysis/
│   ├── Codes/              # Analysis notebooks (plot learning curves, predictions, CFF scans)
│   │   ├── CFF_obs_train_predict_plot.ipynb
│   │   └── Obs_NN_train_predict_plot.ipynb
│   └── Partons_output/     # CSV/JSON output files from fits and predictions
├── libtorch/               # Bundled libtorch installation
├── cmake/Modules/          # Find-modules for external libraries
├── CMakeLists.txt
├── CLAUDE.md               # Developer notes for Claude Code
└── README.md               # This file
```

The `NNFit/Theory/` layout mirrors PARTONS' own `Beans/` and `Modules/` directory hierarchy, with `CFFs`, `Processes`, and `Obs` subdivisions matching the three links of the DVCS calculation chain.

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
Entry point for `Run_CFF_NN_Fit` executable.  Constructs a `CFF_NN_Fitter` with the CLAS15
BSA dataset, runs the workflow:
1. `train_nn()` — train NN with χ² loss on the observable (via `CustomLoss`)
2. `observ_calc()` — compute `DVCSAluMinusSin1Phi` via the PARTONS service using the trained NN
3. `observ_calc_torch()` — compute the same observable through the PARTONS-tensor module chain (direct `computeTensor`)
4. `observ_calc_torch_via_service()` — drive the `*Torch` subclasses through the PARTONS service (verification path)

#### `src/NNFit/CFF_NN_Fit.cpp` — CFF fitter with full observable pipeline
Implements `CFF_NN_Fitter`, the central class of the NNFit subsystem:

- **`train_nn()`** — Adam (lr=1e-2, weight_decay=1e-3), raw (unscaled) inputs, early stopping
  (patience=200, max 10000 epochs) with best-validation parameter restore, writes
  `cff_learning_curve.csv`. Drives observable training via `CustomLoss` (χ² between
  predicted A_LU and the data observable).
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

**DVCS channel layer** (each twin sits next to its scalar counterpart):

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

### Batch computation — PLANNED, NOT YET IMPLEMENTED (branch `6-speedup_via_vectorized_batchobscalc`)

> **Status:** this section describes *future* design only. Batching across data points is
> **not implemented**. The training loop still processes data points **one at a time**
> (`CustomLoss::forward` loops the rows and calls `computeSingleKinematicTorch` per point).
> The two speedups actually done so far are listed under **"Speedup work completed"** below.

The GL swap (done) is the **prerequisite for vectorizing across data points**.  DEXP's
refinement level is chosen adaptively *per kinematic point*, so it cannot fit a single
static `[N, M]` φ grid; a fixed GL rule gives the **same M φ nodes for every point**, which
is exactly what batching would need.

The (unimplemented) planned design would push the batch dimension **down into the existing
reusable layers** (rather than a monolithic batch entry point):

- kinematics carried as `[N]` tensors, one `[N,3]` NN forward, cross-sections as `[N,M]`,
  the φ-integral reduced over the shared grid;
- a batched sibling method on each module (alongside the scalar virtual and the
  single-kinematic tensor method) plus a `computeManyKinematicTorch` service driver,
  mirroring the scalar `computeSingleKinematic ↔ computeManyKinematic` pair;
- adding a new observable then means writing only its thin batched leaf — everything below
  is reused.

It would be **data parallelism**: the multicore speedup would come for free from ATen's
intra-op threading on the batched tensor ops, the graph built once with a **single
deterministic `backward()`**, and the path GPU-ready — without the gradient-accumulation
races of a hand-threaded per-point loop.  The work would be purely additive: the scalar
virtuals and the single-kinematic tensor chain untouched.

---

### Speedup work completed (vs. planned)

To be unambiguous about what is actually in the code:

| Speedup | Status | What it does |
|---|---|---|
| **Fixed GL-10 integrator** (was DEXP) | ✅ **done** | One batched φ-integrand evaluation instead of DEXP's adaptive `L+1` levels (per A_LU computation) |
| **Hoist setup — #1** (prepare/assemble split) | ✅ **done** | The φ- and helicity-independent kinematic factors + CFFs are computed **once per kinematic point** instead of once per beam helicity (twice) |
| **Batch across data points (#3 vectorization)** | ❌ **not implemented** | Would replace the per-point loop with `[N,…]` batched tensor ops (design above) |

So, concretely, the speedup achieved to date is: (1) the **integrator change**
(DEXP → fixed 10-point Gauss–Legendre), and (2) computing the **φ- and helicity-independent
kinematic factors once per kinematic point** (#1). Per-data-point batching is still future
work; training remains a serial loop over the data points.

---

### Hoist setup — prepare/assemble split (#1, 2026-06-24)

A standalone per-point speedup that also de-risks the batched work above.  The torch
asymmetry `DVCSAluMinusTorch::aLUTensor` evaluates the cross section for **both** beam
helicities (λ=±1) to form `A_LU = (σ⁺−σ⁻)/(σ⁺+σ⁻)`.  Previously each call re-ran the whole
φ-independent setup — **one NN forward + the BMJ12 kinematic block + 72 angular
coefficients** — so that helicity-independent work executed **twice per data point**.  (The
scalar PARTONS path has the same two-helicity structure but dodges the cost via CFF caching;
the torch path can't cache without severing the autograd graph, so it hoists instead.)

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

Preparatory changes for the planned CFF-uncertainty replica ensemble (train N replicas on
Monte-Carlo-fluctuated data, band = mean ± σ in Python), plus training quality-of-life fixes:

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

| File | Content | Written by |
|---|---|---|
| `cff_learning_curve.csv` | epoch, train χ², val χ² (every 2 epochs) | `train_nn()` |
| `obs_prediction.csv` | `xB,t,Q2,E,phi,obs_true,obs_pred,error` per point | `predict()` |
| `obs_model_eval.csv` | `observable,mse,r_squared,chi2` | `predict()` |
| `cff_model.json` | trained NN export (`arch`, `dtype`, `best_val_chi2`, `input_features`, `output_layer`, min-max `scaling`, `fc1`/`fc2` weights+biases) — reproduces the exact forward in Python | `predict()` |
| `dvcs_DVCSAluSinPhi_BSACLAS15_ANN.csv` | Mean ± σ observable per kinematic point (replica ensemble, from `ObsCalc_CFFNNReplicas`) | `ObsCalc_CFFNNReplicas` |
| `dvcs_DVCSAluSinPhi_ANN_replicas.csv` | Individual replica values for every kinematic point (from `ObsCalc_CFFNNReplicas`) | `ObsCalc_CFFNNReplicas` |

---

## Data format

Input files are pipe-separated (`|`).  Columns: `xB | t | Q2 | ... | CFF values ... | error`.
Features are the first 3 columns `(xB, t, Q²)`; CFF labels are matched by column name from
the `output_layer` argument; the last column (`error`) is read as σ but not yet used in the
loss function.