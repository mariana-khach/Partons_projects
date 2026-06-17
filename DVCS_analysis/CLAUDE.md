# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

The project uses CMake with two build directories:

- `build/` — Makefile-based build (manual)
- `cmake-build-debug/` — CLion/Ninja build

To build from `build/`:
```bash
cd build && cmake .. && make -j$(nproc)
```

To build a single target (e.g., `Run_CFF_NN_Fit`):
```bash
cd build && make Run_CFF_NN_Fit
```

All executables are placed in `bin/`.

**PARTONS must be run from the `bin/` directory** so it can locate `partons.properties` and `logger.properties`:
```bash
cd bin && ./Run_CFF_NN_Fit
```

## Dependencies

External: ElementaryUtils, NumA++, PARTONS, SFML, CLN, GSL, Apfel++, LHAPDF, libxml2 — all found via `cmake/Modules/`.

libtorch is bundled locally at `libtorch/` and found via `CMAKE_PREFIX_PATH`.

## Executables and their source files

| Executable | Entry point | Purpose |
|---|---|---|
| `DVCS_analysis` | `src/main.cpp` | PARTONS XML-scenario runner + custom obs computations |
| `Run_CFF_NN_Fit` | `src/Run_CFF_NN_Fit.cpp` | Train NN on CFFs → predict → evaluate DVCS observable |
| `NN_CFF_fit` | `src/NN_CFF_fit.cpp` | Earlier/simpler NN fit (no PARTONS observable pipeline) |
| `ObsCalc_CFFNNReplicas` | `src/ObsCalc_CFFNNReplicas.cpp` | Observable calculation using PARTONS replica CFF modules |
| `nn_bsa` | `src/nn_bsa.cpp` | Standalone libtorch NN trained on CLAS15 BSA data |
| `dcgan` | `src/dcgan.cpp` | libtorch smoke test (DCGAN) |

## NNFit subsystem (`src/NNFit/`, `include/NNFit/`)

This is the active development area. It implements a differentiable pipeline:

**NN architecture** (`CFF_NN_Fit.h`): `CFFNNModel` — 3 inputs (xB, t, Q²) → 6 hidden (Tanh) → n outputs. Input features are min-max scaled (fit on training set, applied to both splits). Up to 8 CFF outputs: `{ReH, ImH, ReE, ImE, ReHt, ImHt, ReEt, ImEt}`.

**`CFF_NN_Fitter`** (`CFF_NN_Fit.cpp`) — orchestrates the full workflow:
1. `train_nn()` — Adam optimizer, lr=1e-4, weight_decay=1e-3, early stopping (patience=200, max 10000 epochs). Uses `CustomLoss` (χ² on the observable, see 2026-06-02 session notes). Writes `cff_learning_curve.csv`.
2. `observ_calc()` — plugs the trained model into PARTONS via `DVCSCFFNNTorch` and calls PARTONS service to compute `DVCSAluMinusSin1Phi`
3. `observ_calc_torch()` — computes the same observable through the PARTONS-tensor module chain via `DVCSAluMinusSin1PhiTorch::computeTensor()`, staying inside the libtorch autograd graph
4. `observ_calc_torch_scalar()` — instantiates the `*Torch` subclasses but drives them through PARTONS' standard `DVCSObservableService`; the inherited scalar virtuals wrap the tensor methods under `NoGradGuard`+`.item()` (verification path — same torch physics, gradient dropped)

**`DVCSCFFNNTorch`** (`src/NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.cpp`) — a PARTONS `DVCSConvolCoeffFunctionModule` that wraps `CFFNNModel`. Registered via `BaseObjectRegistry`. Receives kinematics from PARTONS as `(m_xi, m_t, m_Q2)`, converts xB = 2ξ/(1+ξ), applies the training-set min-max scaling carried in via `setModel()`, runs inference, and returns `std::complex<double>` CFF values (scalar) or grad-tracked 0-d complex tensors (`computeCFFTensor`/`computeAllCFFsTensor`).

### Theory submodule (`src/NNFit/Theory/`, `include/NNFit/Theory/`)

A **fully differentiable** DVCS observable chain that runs *inside* the PARTONS module framework (not bypassing it). The design goal is a tensor chain **structurally identical, link-for-link, to PARTONS' scalar chain**: every scalar link has a torch twin with the same role, so gradients (∂A_LU/∂NN-weights) flow end-to-end while the same classes remain drop-in for the scalar pipeline. This enables training directly on observable data.

**Generic, channel-agnostic templates** (`Theory/Modules/…`, header-only — tensor twins of PARTONS' `Observable<K,R>` / `ProcessModule<K,R>` / `ObservableService<K,R>`; the `ResultType` parameter collapses to `torch::Tensor`, so only `KinematicType` is templated):

- **`ObservableTorch<K>`** (`Modules/Obs/ObservableTorch.h`) — NVI idiom mirroring scalar `compute`/`computeObservable`: public template method `computeTensor()` delegates to the protected pure-virtual hook `computeTensorImpl()`.
- **`ProcessModuleTorch<K>`** (`Modules/Processes/ProcessModuleTorch.h`) — channel-agnostic skeleton; no cross-section API (that's channel-specific, as in scalar).
- **`ObservableServiceTorch<K>`** (`Modules/Services/ObservableServiceTorch.h`) — generic driver `computeSingleKinematicTorch(kin, ObservableTorch<K>*)` returning the live tensor (no detach). A **mixin**, not a base, since it's layered onto the existing PARTONS service.

**DVCS channel layer** (`Theory/Modules/…/DVCS/`):

- **`DVCSObservableTorch`** = `ObservableTorch<DVCSObservableKinematic>` (alias).
- **`DVCSProcessModuleTorch`** — tensor twin of `DVCSProcessModule`. Declares the three sub-process atoms `crossSectionBHTensor`/`crossSectionVCSTensor`/`crossSectionInterfTensor` (pure virtual, siblings of `CrossSectionBH/VCS/Interf`) plus the template method `crossSectionTensor(λ, charge, kin, φ, VCSSubProcessType=ALL)` that runs the φ-independent setup once and sums the selected sub-processes — mirroring `DVCSProcessModule::compute = Σ CrossSection*`. `setupKinematicsTorch` is the protected setup hook.
- **`DVCSObservableServiceTorch`** (`Modules/Services/DVCS/`) — `public PARTONS::DVCSObservableService, public ObservableServiceTorch<DVCSObservableKinematic>`. Inheriting the scalar service puts it on the `ServiceObject` branch (registrable, retrievable, full scalar machinery reused); the mixin adds the tensor driver. Self-registers via `BaseObjectRegistry`; fetched by name through `ServiceObjectRegistry::get("DVCSObservableServiceTorch")`.

**Per-class-parallel observable leaves** (mirror scalar `DVCSAluMinus` → `DVCSAluMinusSin1Phi`):

- **`DVCSAluMinusTorch`** (`Modules/Obs/DVCS/`) — `public PARTONS::DVCSAluMinus, public DVCSObservableTorch`. Owns the reusable pointwise asymmetry `aLUTensor(kin, φ[N]) = (σ⁺−σ⁻)/(σ⁺+σ⁻)` (cross-casts `m_pProcessModule` to `DVCSProcessModuleTorch*`); its `computeTensorImpl` returns A_LU at the kinematic's φ. Scalar `computeObservable` wraps `computeTensor().item()`.
- **`DVCSAluMinusSin1PhiTorch`** — `public DVCSAluMinusTorch, public MathIntegratorModuleTorch`. `computeTensorImpl` = the sin(1φ) Fourier moment of the inherited `aLUTensor` via `integrateTorch` (DEXP). No diamond (single path to `PARTONS::DVCSAluMinus`; the integrator is a pure mixin). A future `DVCSAluMinusCos0PhiTorch` derives the same way and reuses `aLUTensor`.

**`DVCSProcessBMJ12Torch`** (`Modules/Processes/DVCS/`) — `public PARTONS::DVCSProcessBMJ12, public DVCSProcessModuleTorch`. Overrides the three sub-process atoms (BMJ12 in `float64` tensors; pure kinematics as cached doubles, CFF-bilinear/linear layers in-graph) + `setupKinematicsTorch` (φ-independent BMJ12 quantities, 72 angular coeffs, one NN forward caching the CFF tensors). Unpolarized target only.

**`MathIntegratorModuleTorch`** (`Modules/`) — tensor twin of `MathIntegratorModule`. `integrateTorch()` evaluates a batched `[N]`-φ integrand and returns the integral in-graph. Supports the **same DEXP** integrator the scalar `DVCSAluMinusSin1Phi` uses (plus TRAPEZOIDAL/GL); `GK21_ADAPTIVE` is unsupported (non-constant-weight extrapolation).

**Chain correspondence** (all base-typed pointers + virtual dispatch, same as scalar):

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

The only irreducible differences: the torch chain returns a grad-carrying `torch::Tensor` (scalar returns a detached `double` via the result bean), and evaluates the BMJ12 formulae + φ-nodes as `float64` tensors/batches (scalar uses native `double` + a scalar φ loop). With the same DEXP and `float64`, all three `observ_calc*` paths agree to every printed digit.

### ⚠️ Coverage caveat — torch path is unpolarized-target only

The torch BMJ12 port (`DVCSProcessBMJ12Torch`) implements **only the unpolarized-target sector** (Λ=0): only the unpolarized BH Fourier coeffs and the S=0 interference angular blocks (`m_Cang`/`m_Sang`) are transcribed; the LP/TP (`dC`/`dS`) rows are omitted. This is valid for A_LU (the only observable ported), whose target is unpolarized.

The hazard is **which scalar physics actually runs depends on which classes you wire**, because the torch leaves' scalar virtuals wrap the tensor methods:

- **Base PARTONS classes** (`DVCSProcessBMJ12` + `DVCSAluMinusSin1Phi`, as in `observ_calc()`) → PARTONS' native arithmetic → **full coverage** (polarized targets included). Use these for any polarized-target observable.
- **Torch leaves driven through the scalar service** (`observ_calc_torch_scalar()`): `DVCSAluMinusTorch::computeObservable` wraps `computeTensor().item()`, so the call routes into the **torch** chain (`crossSectionTensor`), i.e. unpolarized-only physics — *not* the inherited native `CrossSectionBH/VCS/Interf`.

So: **do not wire a `*Torch` observable/process leaf for a polarized-target observable and expect correct results** — its scalar path silently runs the unpolarized torch port (wrong/zero for the polarized contributions), with no error raised. For polarized-target work, use the base PARTONS classes (which remain registered alongside the torch ones). The torch leaves are correct only for unpolarized-target observables (A_LU and siblings). Removing this caveat requires porting the LP/TP coefficient rows into `setupKinematicsTorch` + the sub-process tensor methods.

(Note: `DVCSProcessBMJ12Torch` only overrides the *tensor* sub-process methods `crossSectionBH/VCS/InterfTensor`; it inherits PARTONS' scalar `CrossSectionBH/VCS/Interf` unchanged. So a torch *process* wired under a *base* `DVCSAluMinusSin1Phi` observable would still run full-coverage native scalar physics — the unpolarized restriction only bites along the tensor path, reached via a torch *observable* leaf.)

## Data format

Input CSVs are pipe-separated (`|`). Columns: `xB | t | Q2 | ... | ImH | ReH | ... | error`. Features are the first 3 columns; labels are matched by name from `m_output_layer`; the last column (`error`) is loaded as σ but not currently used in the MSE loss.

Data path and output paths are hardcoded absolute paths in `src/Run_CFF_NN_Fit.cpp` and `src/NNFit/CFF_NN_Fit.cpp` (pointing to `My_Analysis/Partons_output/`). Update these when moving environments.

## Output files (`My_Analysis/Partons_output/`)

| File | Content |
|---|---|
| `cff_learning_curve.csv` | epoch, train_loss, val_loss (written every 2 epochs) |

## PARTONS runtime configuration

`bin/partons.properties` sets log file path, XML schema path, and thread count (`computation.nb.processor`). PARTONS is a singleton: `Partons::getInstance()->init(argc, argv)` / `pPartons->close()`. All PARTONS exceptions are `ElemUtils::CustomException`.

---

## Session notes (2026-04-24)

### Work completed

**Built the theory submodule** (`src/NNFit/theory/`, `include/NNFit/theory/`) — a fully differentiable reimplementation of `DVCSAluMinusSin1Phi` that bypasses PARTONS and stays inside the libtorch autograd graph. Three files:

1. **`DVCSKinematicsTorch`** — pure-C++ precomputation of all φ-independent kinematics from BMJ12 (arXiv:1212.6674): ε, K, K̃, lepton propagator decomposition (P1 = P1_c + P1_K·cos φ, P1+P2 = const), dipole form factors F1/F2, BH Fourier coefficients `cBH[3]`, and the full 3×3×4 angular coefficient arrays `C_ang` and `S_ang` for the interference term.

2. **`DVCSAmplitudesBMJ12Torch`** — torch-tensor physics: `DressedCFFs` (j=0,1,2 helicity combinations), `VCSCoeffs` (purely real for unpolarised target with CFF_FT=0), `InterfCoeffs` (linear in Re/Im CFFs), and `crossSectionAtPhi()` returning a 0-d in-graph tensor.

3. **`DVCSAluMinusSin1PhiTorch`** — 10-point Gauss–Legendre quadrature over φ ∈ [0,2π] using nodes/weights hardcoded from `scipy.special.p_roots(10)`. `compute()` runs a scalar loop of 10 GL nodes (20 `crossSectionAtPhi` calls total). Returns a 0-d `torch::Tensor` carrying ∂A_LU/∂(NN weights).

**Added `CFF_NN_Fitter::observ_calc_torch()`** in `CFF_NN_Fit.cpp` — calls `DVCSAluMinusSin1PhiTorch` at the same kinematics as `observ_calc()` (xB=0.2, t=−0.2, Q2=2, E=5.932) and prints the result.

### Bugs fixed

| Bug | Location | Symptom | Fix |
|---|---|---|---|
| Angular coefficient transcription | `DVCSKinematicsTorch.cpp`, `C_ang[2][0][1]` | tpQ2 applied as extra multiplicand outside the bracket | Moved tpQ2 inside the inner kinematic factor |
| BH prefactor wrong epsroot power | `DVCSAmplitudesBMJ12Torch.cpp`, `crossSectionAtPhi()` | torch result ~3.3% low vs PARTONS (0.00601605 vs 0.00622424) | Changed `kin.er[2]` → `kin.er[3]` (`(1+ε²)²` not `(1+ε²)^{3/2}`) in `prefBH` |

After both fixes the two methods agree.

### Known limitations / open tasks

- **`compute()` is not vectorized**: the GL loop is a scalar 10-iteration C++ `for`. A batched implementation (GL nodes as `[10]` tensors, all φ evaluated simultaneously) would reduce the 20 `crossSectionAtPhi` calls to 2 batch calls. Not yet implemented.
- **`C_ang`/`S_ang` cross-check**: numerical validation of all 72 angular coefficients against PARTONS at a test kinematic point has not been performed.
- **`ObsCalc_CFFNNReplicas.cpp`**: `out1` (`dvcs_DVCSAluSinPhi_ANN_replicas.csv`) captures per-replica values only for `j==3` (the 4th kinematic point from the CLAS15 input file); all other points get only mean ± σ in `out`.

---

## Session notes (2026-05-20)

### Goal

Rework the differentiable observable pipeline so that it lives **inside** the PARTONS module framework rather than bypassing it. Each link of the chain (CFFs → cross section → asymmetry) gets a PARTONS-registered subclass that adds a tensor-returning sibling method alongside the existing scalar PARTONS API. Both pipelines stay in sync by construction, and PARTONS' factory/registry can drive either path.

### What changed

**Step 1 — `DVCSCFFNNTorch::computeCFFTensor(gpdType)`**

Made the NN forward pass the single source of truth.
- New method: `std::pair<torch::Tensor, torch::Tensor> computeCFFTensor(PARTONS::GPDType::Type)` returning the 0-d (Re, Im) tensors connected to the autograd graph. No `NoGradGuard`, no `eval()` toggle — caller controls mode.
- `computeCFF()` is now a thin wrapper: `NoGradGuard` + `eval()` + `computeCFFTensor(m_currentGPDComputeType)` + `.item<float>()` → `std::complex<double>` for PARTONS.

**Step 2 — `DVCSProcessBMJ12Torch` (new PARTONS module)**

PARTONS-registered subclass of `DVCSProcessBMJ12` that exposes the differentiable cross-section.
- New method: `torch::Tensor crossSectionAtPhiTensor(double phi, double beamHelicity)` returns the total σ(λ,φ) = σ_BH + σ_VCS + σ_Interf at a single φ as a 0-d tensor.
- Implementation: `dynamic_cast`s `m_pConvolCoeffFunctionModule` to `DVCSCFFNNTorch*`, pulls the 8 leading-twist CFF tensors via `computeCFFTensor(type)` (one call per H, E, Ht, Et), then chains the existing `Theory::computeDressedCFFs / computeVCSCoeffs / computeInterfCoeffs / crossSectionAtPhi` primitives.
- `buildTorchKinematics()` lazily constructs `Theory::DVCSKin` from the inherited (`m_xB`, `m_t`, `m_Q2`, `m_E`) and caches it so φ-scans don't re-do the kinematic setup.
- Inherited scalar `CrossSectionBH/VCS/Interf` keep working — PARTONS can still drive this module through the normal pipeline.

**Step 3 — `DVCSAluMinusSin1PhiTorch` (new PARTONS module, replaces standalone class)**

Replaced the old `Theory::DVCSAluMinusSin1PhiTorch` standalone driver with a PARTONS-registered subclass of `PARTONS::DVCSAluMinusSin1Phi`.
- New method: `torch::Tensor computeTensor(const DVCSObservableKinematic&)` returns A_LU^{sin1φ} as a 0-d tensor with gradients to the NN weights.
- Implementation: triggers the parent's scalar `compute()` once to push (xB, t, Q², E) onto the process module (return value discarded), then runs a 10-pt Gauss–Legendre quadrature, calling `DVCSProcessBMJ12Torch::crossSectionAtPhiTensor()` at each φ node for both helicities.
- GL nodes/weights moved into this class as `static const` arrays.
- Old `Theory::DVCSAluMinusSin1PhiTorch` files **deleted**; same paths reused for the new class.

**`CFF_NN_Fitter::observ_calc_torch()` migration**

Now mirrors `observ_calc()` exactly, using the PARTONS factory pattern via `BaseObjectRegistry`. All three `*Torch` modules instantiated through `getModuleObjectFactory()`, wired up via `setProcessModule` / `setConvolCoeffFunctionModule`, and driven by `pTorchObs->computeTensor(dvcsKinematics)`.

### Directory restructure

Source/header tree migrated to a PARTONS-style layout:

| File | New location |
|---|---|
| `DVCSCFFNNTorch.{h,cpp}` | `NNFit/theory/Modules/CFFs/DVCS/` |
| `DVCSKinematicsTorch.{h,cpp}` | `NNFit/theory/Beans/Obs/DVCS/` |
| `DVCSAmplitudesBMJ12Torch.{h,cpp}` | `NNFit/theory/Modules/Processes/DVCS/` |
| `DVCSProcessBMJ12Torch.{h,cpp}` | `NNFit/theory/Modules/Processes/DVCS/` (new) |
| `DVCSAluMinusSin1PhiTorch.{h,cpp}` | `NNFit/theory/Modules/Obs/DVCS/` (rewritten) |

All `#include` paths converted from broken relative forms (`"../../include/..."`, `"DVCSKinematicsTorch.h"`) to project-relative form (`"NNFit/..."`) since `include/` is on the include path via `target_include_directories` for `Run_CFF_NN_Fit`. CMakeLists.txt updated in matching `list(REMOVE_ITEM ...)` and `add_executable(Run_CFF_NN_Fit ...)` blocks.

### Module chain after refactor

All PARTONS-registered, each link exposes a tensor sibling:

```
DVCSAluMinusSin1PhiTorch   (computeTensor)       — 10-pt GL quadrature over φ
        ↓ m_pProcessModule
DVCSProcessBMJ12Torch      (crossSectionAtPhiTensor) — σ(λ,φ) tensor
        ↓ m_pConvolCoeffFunctionModule
DVCSCFFNNTorch           (computeCFFTensor)    — NN CFFs as 0-d tensors
```

PARTONS still dispatches normally through the inherited scalar methods on all three. The autograd graph survives end-to-end from NN weights to the final asymmetry along the tensor sibling path. The hard boundary (`DVCSObservableService::computeSingleKinematic()`'s scalar return type) is hit *only* by the scalar path; the differentiable path bypasses the service and calls `computeTensor()` directly on the registered observable subclass.

### Numerical verification

Both paths evaluated at xB=0.2, t=-0.2, Q²=2, E=5.932:
- PARTONS scalar `observ_calc()`:        **-0.00131307**
- PARTONS-tensor `observ_calc_torch()`:  **-0.00131306**

~1-in-6th-sig-fig difference is the quadrature method (PARTONS' adaptive `MathIntegratorModule` vs. fixed 10-pt Gauss–Legendre), not a physics difference.

### Why `DVCSAmplitudesBMJ12Torch` is kept (not folded into the process module)

Pure-physics primitives (`Theory::DressedCFFs`, `Theory::VCSCoeffs`, `Theory::InterfCoeffs`, plus their compute functions and `crossSectionAtPhi`) live separately from the PARTONS orchestration in `DVCSProcessBMJ12Torch`. This keeps the BMJ12 formulae unit-testable without a PARTONS instance and lets a future `DVCSProcess*Torch` subclass reuse the same primitives.

### Known limitations / open tasks (carried forward)

- **GL loop still scalar**: 10 nodes × 2 helicities = 20 sequential `crossSectionAtPhi` calls per asymmetry. A batched implementation (all φ evaluated simultaneously as `[10]` tensors) is still not implemented.
### Resolved limitations (carried forward from 2026-05-20 → 2026-05-22)

- **One-shot kinematics setup wastes a scalar cross-section** — *fixed 2026-05-22*. `DVCSProcessBMJ12Torch::setupKinematics(const DVCSObservableKinematic&)` added (public): calls the inherited protected `setKinematics()` to set (xB, t, Q², E, φ) on the parent's members, runs the xi converter to push `(xi, t, Q²)` onto the attached `DVCSCFFNNTorch` via a new `setupKinematics(xi, t, Q2)` helper, and refreshes the cached `Theory::DVCSKin` — no scalar pipeline runs. `DVCSAluMinusSin1PhiTorch::computeTensor()` now calls this helper instead of `pProc->compute(...)`. Saves 4 NoGrad NN forwards plus one full BH+VCS+Interf scalar arithmetic block per data point per loss call.
- **Redundant per-GPD-type forward passes** — *fixed 2026-05-22*. The previous `DVCSProcessBMJ12Torch::crossSectionAtPhiTensor(phi, helicity)` re-did the four-CFF NN-forward + dressed-CFF + Fourier-coefficient build on every call, so the GL quadrature in the observable was paying that cost 20× per asymmetry. Refactored:
  - `DVCSCFFNNTorch::computeAllCFFsTensor()` — single NN forward, returns an `AllCFFsTensor` struct with all eight CFF components.
  - `DVCSProcessBMJ12Torch::computeFourierCoeffsTensor()` — runs the NN forward once via `computeAllCFFsTensor`, builds `Theory::DressedCFFs`, returns a `std::pair<Theory::VCSCoeffs, Theory::InterfCoeffs>`.
  - `DVCSProcessBMJ12Torch::crossSectionAtPhiTensor(vcs, interf, phi, helicity)` — now lightweight: just the φ-dependent assembly via `Theory::crossSectionAtPhi`.
  - `DVCSAluMinusSin1PhiTorch::computeTensor()` calls `computeFourierCoeffsTensor` once outside the GL loop, then `crossSectionAtPhiTensor(vcs, interf, ...)` 20× inside.
  - Per-data-point cost drops from 80 NN forwards to **1**.
- **`C_ang`/`S_ang` cross-check** and **`ObsCalc_CFFNNReplicas` per-replica capture** issues from the 2026-04-24 session are unchanged.

---

## Session notes (2026-06-02)

### Goal

Use the PARTONS-tensor module chain built in the 2026-05-20 session to train the NN **directly on DVCS observable data** (not on CFF labels). Add a third verification path that drives the `*Torch` subclasses through PARTONS' standard service to prove they remain operable as drop-in replacements.

### What changed

**χ² loss on the observable** — new files `include/NNFit/CustomLoss.h` and `src/NNFit/CustomLoss.cpp`:

- `CustomLossImpl : torch::nn::Module`, wrapped as `TORCH_MODULE(CustomLoss)` so it's instantiated and called like any PyTorch loss module.
- Constructor instantiates the three PARTONS-tensor modules via `BaseObjectRegistry`/`ModuleObjectFactory` once, wires them up (xi-converter, scales, CFF, process, observable), and injects the trained-or-in-training `CFFNNModel` into `DVCSCFFNNTorch::setModel`.
- `forward(X, E, y_obs, sigma)` loops over rows: builds a `DVCSObservableKinematic` per data point, calls `pObs->computeTensor(kin)` (which goes through `setupKinematics` → `computeFourierCoeffsTensor` → 20× lightweight `crossSectionAtPhiTensor`), accumulates `((pred − y_obs[i]) / sigma[i])²` into a single 0-d tensor with autograd connected to every NN parameter.
- Formula:  χ² = Σᵢ ( A_LU^{sin1φ}(NN(xᵢ)) − yᵢ )² / σᵢ²

**Observable-data loader** — `CFF_NN_Fitter::load_data_observable()` added (alongside the existing `load_data()`). Reads the BSA-format CSV `xB|t|Q2|E|phi|DVCSAluSinPhi|error` and returns `(X[N,3], E[N], y_obs[N], sigma[N])`. φ is dropped — the observable integrates it out.

**`train_nn()` rewrite** — now drives observable training:

- Uses `load_data_observable()` (4-tuple) instead of `load_data()` (3-tuple).
- Constructs `CustomLoss(m_net, m_output_layer)` once; the per-epoch loss closure becomes `loss_fn(X_train, E_train, y_obs_train, sigma_train)`.
- **Min-max input scaling removed**: the PARTONS `DVCSCFFNNTorch` module feeds raw `(xB, t, Q²)` to the NN, so applying scaling at training time would have made the trained NN incompatible with the inference pipeline. `m_X_min` / `m_X_max` are set to identity values to keep the rest of the codebase happy.
- Early stopping / learning-curve / train-val split structure unchanged.

**Third verification path** — `CFF_NN_Fitter::observ_calc_torch_via_service()`:

- Mirrors `observ_calc()` wiring exactly except that it instantiates `DVCSProcessBMJ12Torch::classId` and `DVCSAluMinusSin1PhiTorch::classId` (instead of the base PARTONS classes).
- Drives the computation through `DVCSObservableService::computeSingleKinematic()` — so only the **inherited scalar virtuals** (`CrossSectionBH/VCS/Interf` from `DVCSProcessBMJ12`, `MathIntegratorModule`-based φ-integral from `DVCSAluMinusSin1Phi`) run.
- Wired into `Run_CFF_NN_Fit.cpp` `main()` after the existing `observ_calc()` and `observ_calc_torch()` calls.

### Numerical verification (three pipelines, same kinematics)

After a short 5-epoch χ² training run at xB=0.2, t=−0.2, Q²=2, E=5.932:

| Path | Modules involved | Driver | A_LU^{sin1φ} |
|---|---|---|---|
| `observ_calc()`                    | base `DVCSAluMinusSin1Phi` + base `DVCSProcessBMJ12` + `DVCSCFFNNTorch` | PARTONS service | **0.0145146** |
| `observ_calc_torch_via_service()`  | `*Torch` subclasses (scalar virtuals only)                                | PARTONS service | **0.0145146** |
| `observ_calc_torch()`              | `*Torch` subclasses (tensor entry points)                                 | direct `computeTensor` | **0.0145147** |

Rows 1 and 2 agree to all printed digits — empirical proof that the `*Torch` subclasses inherit and respect the PARTONS scalar contract. Row 3 differs in the 6th sig fig from rows 1/2 — the known 10-pt Gauss–Legendre vs. PARTONS adaptive-quadrature precision gap, not a physics difference.

### Caveats

- **NN ignores μF², μR², and QCD-order**: `setQCDOrderType(LO)` and `DVCSScalesQ2Multiplier` are configured and propagated through PARTONS' plumbing, but `DVCSCFFNNTorch::computeCFFTensor` only reads `m_xi, m_t, m_Q2` — `m_MuF2`, `m_MuR2`, `m_qcdOrderType` are vestigial. The NN was trained on 3 features. Switching LO → NLO or changing the Q² multiplier would not change the predicted A_LU. Making the NN scale-aware would require either expanding its input features (then retraining on labels generated at those scales) or routing the NN output through PARTONS' evolution module.
- **χ² training is slow per epoch**: even with the two performance fixes from 2026-05-22, each step is N (data points) × 1 grad-tracked NN forward + N × `computeFourierCoeffsTensor` + N × 20 cross-section assemblies. For the 10-row CLAS15 file this is ~0.2 s/epoch; for an ~80-row dataset expect ~1–2 s/epoch. The next optimisation target is the **scalar GL loop inside `computeTensor`** — replacing it with a batched implementation (all φ evaluated simultaneously as `[10]`-shape tensors) would collapse 20 cross-section calls into 2.

### Known limitations / open tasks (updated)

- **GL loop still scalar** (carried forward — see "Caveats" above for impact analysis).
- **NN architecturally scale-blind** (new — see "Caveats").
- **`C_ang`/`S_ang` numerical cross-check** and **`ObsCalc_CFFNNReplicas` per-replica capture** issues from the 2026-04-24 session are unchanged.

### Resolved this session (cumulative)

- *2026-05-22*: discarded scalar setup (replaced by `setupKinematics`).
- *2026-05-22*: redundant per-GPD-type NN forwards (replaced by `computeAllCFFsTensor` + `computeFourierCoeffsTensor` + split `crossSectionAtPhiTensor`).
- *2026-06-02*: PARTONS-service compatibility of `*Torch` subclasses verified end-to-end (`observ_calc_torch_via_service`).

---

## Session notes (2026-06-15)

### `MathIntegratorModuleTorch` — differentiable φ-integration

New files `include/NNFit/MathIntegratorModuleTorch.h` and `src/NNFit/MathIntegratorModuleTorch.cpp`: a libtorch counterpart of `PARTONS::MathIntegratorModule`, meant to be inherited by a tensor observable (the way `DVCSAluMinusSin1Phi` inherits the scalar `MathIntegratorModule`). It replaces the hard-coded fixed 10-pt Gauss–Legendre φ-loop in `computeTensor` with a configurable, **gradient-preserving** quadrature.

**Interface** (protected, for the inheriting observable):
- `setIntegrator(NumA::IntegratorType1D::Type)` — same selection mechanism as the scalar base; owns its own `NumA::Integrator1D*` (independent of the scalar one the parent already holds — no diamond).
- `integrateTorch(f, a, b)` → `torch::Tensor` — `f` takes the `[N]` node tensor already mapped onto `[a,b]` and returns `[N]`; result is a 0-d tensor with autograd connected through `f`. **Batched** (the chosen callback shape), so it closes the long-standing "GL loop still scalar" task: fixed rules call `f` once, DEXP at most once per refinement level.

**Why gradients survive**: every supported rule reduces to `∫ = Σ wᵢ f(xᵢ)` with nodes/weights that are *constants* (no dependence on NN parameters), so the graph flows entirely through `f(xᵢ)`.

**Supported types** (each matched against the corresponding NumA scalar routine):
- `GL`, `TRAPEZOIDAL` — read `getNodes()`/`getWeights()` from `NumA::QuadratureIntegrator1D` (reference interval), linear remap `x = d + c·node`, one batched eval. Mirrors `QuadratureIntegrator1D::integrate`.
- `TRAPEZOIDALLOG` — log-spaced trapezoid (requires `a,b>0`); `TrapezoidalLogIntegrator1D` *overrides* the linear mapping, so this path is replicated separately as a fixed weighted sum over `xᵢ = exp(logA + i·logStep)`.
- `DEXP` — adaptive tanh-sinh (double exponential). **This is the one that matters**: `DVCSAluMinusSin1Phi` and ~64 other PARTONS observables select `DEXP` as their φ-integrator. NumA hard-codes ~1500 nodes/weights; we **regenerate** them from the closed form `node(t)=tanh(π/2·sinh t)`, `weight(t)=(π/2)·cosh t / cosh²(π/2·sinh t)` (verified identical to NumA's stored table to full double precision) rather than copy the table. The refinement *level* is chosen by NumA's exact convergence test on detached `.item()` values; the running estimate is kept as a tensor. Once the level is fixed, DEXP's accumulation is provably a fixed weighted sum (`Iₗ = c·0.5ᴸ·Σ wᵢ(f⁺+f⁻)`), so the value matches NumA bit-for-bit *and* gradients flow.
- `GK21_ADAPTIVE` — **throws**. Its Wynn epsilon extrapolation (`qelg`) makes the result a nonlinear rational function of parameter-dependent partial sums — not expressible as a constant-weight sum. PARTONS never selects GK21 (0 call sites), so this is harmless.

**Status**: compiles clean against the real build flags (the editor's missing-`-I` diagnostics are cosmetic). Wired into the `Run_CFF_NN_Fit` target in `CMakeLists.txt` and excluded from the GLOB'd main `DVCS_analysis` executable (libtorch). The DEXP and log-trapezoid reformulations were validated numerically (sin²→π, cos→0, Gaussian, x³, ∫dx/x→ln10).

**Not yet done**: not inherited by `DVCSAluMinusSin1PhiTorch` yet, and no C++ cross-check that `integrateTorch(DEXP,…).item()` equals NumA's scalar `DExpIntegrator1D::integrate` on the real φ-integrand (Python validated the math; a C++ assert would confirm dtype/tolerance plumbing). Once wired in with `DEXP`, the tensor path uses the *same* rule as the scalar path, so `observ_calc_torch()` should match `observ_calc()` to precision instead of carrying the 6th-sig-fig GL-vs-adaptive gap.

### Full Torch observable chain reimplemented (`reimplementing_ALU_torchcalc`)

Reimplemented the entire `DVCSAluMinusSin1Phi` observable computation as a parallel Torch path that mirrors the scalar PARTONS class hierarchy, so the NN can train directly on A_LU^{sin1φ} data with gradients flowing through the whole BMJ12 computation. New files under `NNFit/Theory/Modules/`:

```
DVCSObservableServiceTorch : DVCSObservableService          (Services/DVCS/)
    computeSingleKinematicTorch(kin, DVCSAluMinusSin1PhiTorch*) -> torch::Tensor
        ↓
DVCSAluMinusSin1PhiTorch : DVCSAluMinusSin1Phi, MathIntegratorModuleTorch   (Obs/DVCS/)
    computeTensor(kin) = integrateTorch(DEXP, φ→A_LU(φ)·sinφ, 0..2π)/π
        ↓ m_pProcessModule
DVCSProcessBMJ12Torch : DVCSProcessBMJ12                     (Processes/DVCS/)
    setupKinematicsTorch(kin); crossSectionTensor(λ, charge, φ[N])
        ↓ m_pConvolCoeffFunctionModule
DVCSCFFNNTorch : DVCSConvolCoeffFunctionModule              (CFFs/DVCS/, extended)
    computeAllCFFsTensor() / computeCFFTensor(GPDType)
```

**Dual-use (single source = torch)**: each module's scalar virtual wraps the tensor method under `NoGradGuard` + `.item()` — `computeCFF()`→`computeCFFTensor`, `computeObservable()`→`computeTensor`. So the same registered classes serve both the differentiable path and PARTONS' scalar pipeline.

**Service registration without patching PARTONS**: `DVCSObservableServiceTorch` self-registers in the `BaseObjectRegistry` (protected-ctor idiom, same as the scalar service) and is fetched by name via `getServiceObjectRegistry()->get("DVCSObservableServiceTorch")` + `static_cast` — `ServiceObjectRegistry::get(string)` just delegates to `BaseObjectRegistry::get`. No PARTONS library files modified.

**Process port — surgical, exploiting two facts**: (1) the NN provides only H,E,Ht,Et, so transversity (`CFF_FT`) and twist-3 (`CFF_FLT`) are zero and `m_CFF[i][j] = cF[j][0]·CFF_std[i]`; (2) A_LU has an **unpolarized target** (`Vector3D(0,0,0)` → Λ=0), killing all LP/TP coefficient rows. So only the **unpolarized** sector is ported. The pure-kinematic BMJ12 machinery (`computeAngularCoeffsInterf` m_C/m_S blocks, BH Fourier coeffs, K/ε/form-factors/phase-space, cF) is transcribed **verbatim as doubles** (no grad) into `setupKinematicsTorch`; only the CFF layers are tensors — `C_VCS0` (bilinear), `C_I0`/`C_I0n`/`S_I0n` (linear). Base `DVCSProcessModule` members it reuses are **protected** (`m_xB,m_t,m_Q2,m_E,m_y,m_epsilon,m_tmin,m_tmax,m_pConvolCoeffFunctionModule`, `setKinematics`); the BMJ12-specific kinematics are private, hence the verbatim re-derivation. `crossSectionTensor` assembles `phaseSpace·[A_BH·BH + A_VCS·VCS + A_Interf·I]` batched over the φ quadrature nodes (BH is pure-kinematic/no-grad; P1,P2 and the cos/sin(nφ₁) harmonics are the φ-dependent factors).

**Verification** — three paths in `CFF_NN_Fit.cpp`, all at xB=0.2, t=−0.2, Q²=2, E=5.932 after a training run (the absolute value depends on the freshly-seeded NN; what matters is cross-path agreement):

| Path | Modules | Driver | A_LU^{sin1φ} |
|---|---|---|---|
| `observ_calc()` | base `DVCSAluMinusSin1Phi` + base `DVCSProcessBMJ12` + `DVCSCFFNNTorch` | scalar service `computeSingleKinematic` | **0.00560125** |
| `observ_calc_torch_scalar()` | `*Torch` subclasses (scalar virtuals) | scalar service `computeSingleKinematic` | **0.00560125** (no grad) |
| `observ_calc_torch()` | `*Torch` subclasses (tensor entry points) | `DVCSObservableServiceTorch::computeSingleKinematicTorch` | **0.00560125**, `requires_grad = true` |

All three agree to every printed digit. The two cross-checks: **base scalar vs Torch scalar** proves the `*Torch` modules honor the scalar drop-in contract under PARTONS' normal machinery (their scalar virtuals wrap the tensor methods under `NoGradGuard`+`.item()`); **Torch scalar vs Torch tensor** proves the gradient-preserving path computes the same value, just with the autograd graph attached. The old GL-vs-adaptive 6th-sig-fig gap is gone because the tensor path integrates φ with DEXP, the same rule as the scalar path.

**Caveats / not yet done**: only the unpolarized BMJ12 sector is ported (polarized-target observables need the LP/TP `dC`/`dS` rows). `setupKinematicsTorch` re-derives kinematics every call (one NN forward + the verbatim kinematic block) — the per-φ scalar waste noted in earlier sessions is avoided, but a `computeFourierCoeffsTensor`-once refactor across the two helicity calls is still possible. `CustomLoss` not yet rewired to drive through `DVCSObservableServiceTorch` (it calls `computeTensor` directly, which is equivalent).

---

## Session notes (2026-06-16)

### Goal

Make the Torch chain **generic (channel-ready) and link-for-link symmetric with the scalar chain**, so a future TCS-torch / DVMP-torch needs only new channel leaves, and every scalar method has a same-shaped torch twin. See the rewritten "Theory submodule" section above for the resulting structure.

### What changed

**Rename `DVCSCFFNNPytorch` → `DVCSCFFNNTorch`** throughout (class, files, registry string, includes, CMake, docs). The lone remaining "Pytorch" is the library-reference exception message in `setModel`.

**Min-max scaling carried into the CFF module.** `train_nn()` fits `m_X_min`/`m_X_max` on the training set; these are now passed into `DVCSCFFNNTorch::setModel(net, outputLayer, xMin={}, xMax={})` (folded into `setModel`, not a separate `setScaling`, so the model and its preprocessing travel together). `computeCFF`/`computeCFFTensor` apply `(x−xMin)/(xMax−xMin)` (guarded by `m_xMin.defined()`, so undefined = raw features). Fixes a real train/inference scaling mismatch.

**Generic templates introduced** (`ObservableTorch<K>`, `ProcessModuleTorch<K>`, `ObservableServiceTorch<K>`), instantiated for DVCS via alias / channel subclass — tensor twins of PARTONS' `Observable<K,R>` / `ProcessModule<K,R>` / `ObservableService<K,R>`. `ResultType` collapses to `torch::Tensor`.

**NVI symmetry on the observable.** `ObservableTorch::computeTensor()` is now a public template method delegating to the protected pure-virtual `computeTensorImpl()` — the exact analog of scalar `compute()`/`computeObservable()`. Leaves override `computeTensorImpl`.

**Per-class parallelism on the observable.** New `DVCSAluMinusTorch` (pointwise asymmetry, owns reusable `aLUTensor`); `DVCSAluMinusSin1PhiTorch` now derives from it (mirroring scalar `DVCSAluMinusSin1Phi : DVCSAluMinus`) and reuses `aLUTensor`. A future `DVCSAluMinusCos0PhiTorch` plugs in the same way.

**Per-sub-process granularity on the process.** `DVCSProcessModuleTorch` now declares `crossSectionBHTensor`/`crossSectionVCSTensor`/`crossSectionInterfTensor` (pure virtual, twins of `CrossSectionBH/VCS/Interf`) and a template-method `crossSectionTensor(λ, charge, kin, φ, VCSSubProcessType=ALL)` that runs `setupKinematicsTorch` once and sums the selected sub-processes — mirroring `DVCSProcessModule::compute = Σ CrossSection*` with the same `VCSSubProcessType` selector. The monolithic `crossSectionTensor` in `DVCSProcessBMJ12Torch` was split into the three component overrides (former `sigmaBH/sigmaVCS/sigmaI` locals); `setupKinematicsTorch` is now a private override taking `kin` (no public setup; the cross-section call is self-contained like scalar `compute`).

**Service is generic.** `DVCSObservableServiceTorch` now mixes in `ObservableServiceTorch<DVCSObservableKinematic>` (the templated driver) atop `PARTONS::DVCSObservableService`; `computeSingleKinematicTorch` takes a base `DVCSObservableTorch*` (was the leaf type), so it drives any DVCS tensor observable polymorphically.

**Header move.** `Theory/Modules/ObservableTorch.h` → `Theory/Modules/Obs/ObservableTorch.h` (alongside the other observable headers).

### Registry / type facts confirmed (for reference)

- All PARTONS objects register through the **one** `BaseObjectRegistry` (`registerBaseObject` + `classId`); modules and services use the identical mechanism. Registration does **not** distinguish service from module.
- The service-vs-module distinction is the **inheritance branch**: `ModuleObject : BaseObject, ElemUtils::Thread` vs `ServiceObject : BaseObject`. `DVCSObservableServiceTorch` is on the `ServiceObject` branch via `DVCSObservableService`.
- `ServiceObjectRegistry` is a **typed façade** over `BaseObjectRegistry` (holds a pointer to it, no storage of its own); `get(name/classId)` delegates and `static_cast`s to `ServiceObject*`. The typed getters (`getDVCSObservableService()`) are hard-coded in PARTONS per built-in service — there is none for our out-of-tree service, hence the by-name `get("DVCSObservableServiceTorch")` + cast.

### Verification

Rebuilds clean (no warnings); all three `observ_calc*` paths still agree to every printed digit, `observ_calc_torch()` keeps `requires_grad = true`. No physics moved — the granularity split only promoted existing locals to overridable methods.

### Open tasks (carried forward)

- `CustomLoss` to be (re)implemented driving through `DVCSObservableServiceTorch::computeSingleKinematicTorch` (decided this session).
- Only the unpolarized BMJ12 sector ported; per-sub-process methods recompute their own φ-harmonics/propagators (matches scalar's independent components; minor duplicate tensor arithmetic).
- `C_ang`/`S_ang` numerical cross-check and `ObsCalc_CFFNNReplicas` per-replica capture (from 2026-04-24) still open.
