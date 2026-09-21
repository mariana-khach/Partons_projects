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

**Run from the project root (`DVCS_analysis/`), not from `bin/`:**
```bash
./bin/Run_CFF_NN_Fit
```
`Partons::init` derives the properties-file directory from `argv[0]` (`Partons.cpp:81–88`), so `bin/partons.properties` is found either way — but the paths *inside* it (`log.file.path = bin/logger.properties`, `xml.schema.file.path = data/xmlSchema.xsd`) resolve against the **actual CWD**. From `bin/` they become `bin/bin/…` and the run fails on `logger.properties`. (Rediscovered 2026-06-22; this section said the opposite until 2026-09-17.)

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

**NN architecture** (`CFF_NN_Fit.h`): `CFFNNModel` — 3 inputs (xB, t, Q²) → 6 hidden (Tanh) → n outputs. Input features are min-max scaled (fit on training set, applied to both splits). Up to 8 CFF outputs: `{ReH, ImH, ReE, ImE, ReHt, ImHt, ReEt, ImEt}`. The NN output can be rescaled by a power of xB before being used as the CFF: `CFF = xB^m_xPow * NNet_output` (`m_xPow` defaults to 0, i.e. no rescaling), applied once inside `DVCSCFFNNTorch::forwardNNBatch()` and threaded through every entry point via `setModel(net, outputLayer, xMin, xMax, xPow)`. See 2026-08-05 session notes (it was introduced in the since-removed single-point `forwardNN()`; the batched forward is the only one left).

**`CFF_NN_Fitter`** (`CFF_NN_Fit.cpp`) — orchestrates the full workflow:
1. `train_nn()` — trains the NN **directly on observable data** via `CustomLoss` (reduced χ²/n on A_LU^{sin1φ} computed through the differentiable `*Torch` chain, driven by `DVCSObservableServiceTorch::computeSingleKinematicTorch`). Adam, lr=1e-2 (no weight decay), early stopping (patience=1000, max 10000 epochs). Loads with `load_data_observable()`; writes `cff_learning_curve.csv`. See 2026-06-18 and 2026-09-01 session notes.
2. `predict()` — evaluates the trained model's **observable** prediction per data point through the same `*Torch` chain and compares to the measured value; writes `obs_prediction.csv` (per-point true/pred) and `obs_model_eval.csv` (MSE, R², reduced χ²/n).
3. `observ_calc()` — base-PARTONS scalar reference: `DVCSCFFNNTorch` + base `DVCSProcessBMJ12` + base `DVCSAluMinusSin1Phi` via `DVCSObservableService`.
4. `observ_calc_torch()` — same observable through the PARTONS-tensor chain via `DVCSObservableServiceTorch::computeSingleKinematicTorch` → `computeTensor()`, inside the libtorch autograd graph.
5. `observ_calc_torch_scalar()` — the `*Torch` subclasses driven through the standard `DVCSObservableService`; the inherited scalar virtuals wrap the tensor methods under `NoGradGuard`+`.item()` (verification path — same torch physics, gradient dropped).
5b. `observ_calc_scalar_cff()` — differential test of the **BMJ12 transcription itself**, with the network out of the picture: fixed CFFs (`DVCSCFFConstant`) are pushed through PARTONS' native scalar process module and, via `DVCSCFFScalarTorch`, through `DVCSProcessBMJ12Torch`. The two sides then share nothing but four constant numbers, so any disagreement is in the arithmetic. Needs no trained model. See 2026-09-21 session notes.
6. `train_replicas(n_replicas, …)` — trains a Monte Carlo replica ensemble for a CFF uncertainty band: each replica independently fits Monte-Carlo-smeared pseudodata (`y_smeared = y_obs + N(0, sigma)`), with a fresh train/val split, fresh weight init, and fresh optimizer per replica, via the shared `fit_once()` helper (also used internally by `train_nn()` for the unsmeared central fit). A replica whose validation loss diverges (NaN/Inf) or stays above a "hopeless" reduced-χ²/n threshold at a periodic checkpoint epoch is discarded and fully redrawn (fresh smear + split + init, not just weight reinit), up to `max_tries_per_replica` **total tries** (default 30 — one initial fit plus up to 29 redraws). If every try is hopeless the ensemble is abandoned: the replicas accepted so far are exported (that compute is not lost), then a `std::runtime_error` is thrown naming the replica, the try count and how many were exported. A short ensemble returned silently would be read downstream as a complete one, so the run fails loudly instead. Populates `m_replicas` (`std::vector<TrainedModel>`, each carrying its own net + min-max scaling + best val loss).
7. `export_replicas(out_dir, name_prefix)` — writes each trained replica as `<name_prefix><NN>.json` (same format as `cff_model.json`, via a `net`/scaling-parameterized overload of `export_model_json()`), for out-of-process (Python) mean ± σ CFF bands. **Deletes any pre-existing `<name_prefix>*.json` in `out_dir` first**, so the directory always describes the run that just finished — otherwise a shorter ensemble (10 replicas, then 5) or an aborted one leaves stale files that a `glob` in the plotting code reads as part of the current set. Nothing is deleted when there is nothing to write, so a run that fails before its first replica leaves the previous ensemble intact. See 2026-09-01 and 2026-09-18 session notes.

**`DVCSCFFModuleTorch`** (`Theory/Modules/CFFs/DVCS/`, header-only) — tensor twin of `PARTONS::DVCSConvolCoeffFunctionModule`, added 2026-09-21 to complete the mixin pattern the other two links always had (`DVCSObservableTorch`, `DVCSProcessModuleTorch`). Owns the `AllCFFsTensorBatch` struct and one pure virtual `computeAllCFFsTensorBatch(xB, t, Q2, E)`. Before it existed the process module cross-cast to the **concrete** `DVCSCFFNNTorch`, pinning the tensor chain to one CFF implementation. `E` is in the signature because an implementation that defers to PARTONS' xi-converter and scales modules needs the full kinematics — the network ignores it.

**`DVCSCFFScalarTorch`** (`Theory/Modules/CFFs/DVCS/`) — presents any scalar `DVCSConvolCoeffFunctionModule` (`DVCSCFFConstant`, `DVCSCFFStandard`, `DVCSCFFDispersionRelation`, …) as a `DVCSCFFModuleTorch`, evaluating it per point and packing the results into `[N]` **no-grad** complex tensors. Kinematics are built exactly as `DVCSProcessModule::computeConvolCoeffFunction` does (ξ from the xi-converter, μF²/μR² from the scales module — the same instances the process is wired with, so it cannot drift from the scalar path). Not a PARTONS module (no `classId`, not factory-created), so it is injected with `DVCSProcessModuleTorch::setCFFModuleTorch()`, which takes precedence over the wired convol-coeff module. Non-owning.

**`DVCSCFFNNTorch`** (`src/NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.cpp`) — a PARTONS `DVCSConvolCoeffFunctionModule` **and** a `DVCSCFFModuleTorch` (dual base, like the other links) that wraps `CFFNNModel`. Registered via `BaseObjectRegistry`. Receives kinematics from PARTONS as `(m_xi, m_t, m_Q2)`, converts xB = 2ξ/(1+ξ), applies the training-set min-max scaling carried in via `setModel()`, runs inference, and returns `std::complex<double>` CFF values (scalar, via `computeCFF`) or grad-tracked complex tensors. The tensor side is batched: `computeAllCFFsTensorBatch(xB, t, Q2)` (all four CFFs, one NN forward, `[N]` each — what the torch chain calls) and `computeCFFTensorBatch(type, …)`; `computeCFFTensor(type)` survives as an N=1 wrapper because the scalar `computeCFF()` needs it one GPD type at a time.

### Theory submodule (`src/NNFit/Theory/`, `include/NNFit/Theory/`)

A **fully differentiable** DVCS observable chain that runs *inside* the PARTONS module framework (not bypassing it). The design goal is a tensor chain **structurally identical, link-for-link, to PARTONS' scalar chain**: every scalar link has a torch twin with the same role, so gradients (∂A_LU/∂NN-weights) flow end-to-end while the same classes remain drop-in for the scalar pipeline. This enables training directly on observable data.

**Generic, channel-agnostic templates** (`Theory/Modules/…`, header-only — tensor twins of PARTONS' `Observable<K,R>` / `ProcessModule<K,R>` / `ObservableService<K,R>`; the `ResultType` parameter collapses to `torch::Tensor`, so only `KinematicType` is templated):

- **`ObservableTorch<K>`** (`Modules/Obs/ObservableTorch.h`) — NVI idiom mirroring scalar `compute`/`computeObservable`: public template method `computeTensor()` delegates to the protected pure-virtual hook `computeTensorImpl()`.
- **`ProcessModuleTorch<K>`** (`Modules/Processes/ProcessModuleTorch.h`) — channel-agnostic skeleton; no cross-section API (that's channel-specific, as in scalar).
- **`ObservableServiceTorch<K>`** (`Modules/Services/ObservableServiceTorch.h`) — generic driver `computeSingleKinematicTorch(kin, ObservableTorch<K>*)` returning the live tensor (no detach). A **mixin**, not a base, since it's layered onto the existing PARTONS service.

**DVCS channel layer** (`Theory/Modules/…/DVCS/`):

- **`DVCSObservableTorch`** = `ObservableTorch<DVCSObservableKinematic>` (alias).
- **`DVCSProcessModuleTorch`** — tensor twin of `DVCSProcessModule`. **Batched API only** (the single-kinematic path was deleted 2026-09-16 — see that session's notes). Declares the three sub-process atoms `crossSectionBHTensorBatch`/`crossSectionVCSTensorBatch`/`crossSectionInterfTensorBatch` (pure virtual, siblings of `CrossSectionBH/VCS/Interf`), each `[N,M]` (N data points × M φ nodes). The cross-section call is split **prepare + assemble**: public `prepareTensorBatch(xB, t, Q2, E)` runs the φ-/helicity-independent `setupKinematicsTorchBatch` once (and sets the `m_preparedBatch` guard); the **lightweight** `crossSectionTensorBatch(λ, charge, φ, VCSSubProcessType=ALL)` overloads assume prepare ran and only sum the selected sub-processes (mirroring `DVCSProcessModule::compute = Σ CrossSection*`). `setupKinematicsTorchBatch` is the protected setup hook. Callers must call prepare immediately before assemble for their own kinematics; the guard only catches "never prepared at all," not a mismatched prepare.
- **`DVCSObservableServiceTorch`** (`Modules/Services/DVCS/`) — `public PARTONS::DVCSObservableService, public ObservableServiceTorch<DVCSObservableKinematic>`. Inheriting the scalar service puts it on the `ServiceObject` branch (registrable, retrievable, full scalar machinery reused); the mixin adds the tensor driver. Self-registers via `BaseObjectRegistry`; fetched by name through `ServiceObjectRegistry::get("DVCSObservableServiceTorch")`.

**Per-class-parallel observable leaves** (mirror scalar `DVCSAluMinus` → `DVCSAluMinusSin1Phi`):

- **`DVCSAluMinusTorch`** (`Modules/Obs/DVCS/`) — `public PARTONS::DVCSAluMinus, public DVCSObservableTorch`. Owns the reusable pointwise asymmetry `aLUTensorBatch(xB, t, Q2, E, φ[M]) = (σ⁺−σ⁻)/(σ⁺+σ⁻)`, `[N,M]` (cross-casts `m_pProcessModule` to `DVCSProcessModuleTorch*`). It calls `prepareTensorBatch` **once**, then the lightweight `crossSectionTensorBatch(±1, −1, φ)` per helicity — so the helicity-independent setup (NN forward + BMJ12 kinematics + 72 coeffs) runs once per batch instead of twice. ⚠️ Its own leaf hooks are **not implemented**: `computeTensorImplBatch` throws (a real pointwise version needs an own-φ `[N]` broadcast — see the 2026-09-15 open task), and `computeTensorImpl` is a thin N=1 wrapper over it, so it throws too. Consequently the inherited scalar `computeObservable` (which wraps `computeTensor().item()`) also throws for a **bare** `DVCSAluMinusTorch` — only its moment subclasses are usable. Nothing instantiates it today.
- **`DVCSAluMinusSin1PhiTorch`** — `public DVCSAluMinusTorch, public MathIntegratorModuleTorch`. The only observable leaf actually wired anywhere. `computeTensorImplBatch` = the sin(1φ) Fourier moment of the inherited `aLUTensorBatch` via `integrateTorchBatch` (fixed GL-10, see 2026-06-22 #2); `computeTensorImpl` is a thin N=1 wrapper around it (wraps the single kinematic into a one-element `List<K>`). No diamond (single path to `PARTONS::DVCSAluMinus`; the integrator is a pure mixin). A future `DVCSAluMinusCos0PhiTorch` derives the same way and reuses `aLUTensorBatch`.

**`DVCSProcessBMJ12Torch`** (`Modules/Processes/DVCS/`) — `public PARTONS::DVCSProcessBMJ12, public DVCSProcessModuleTorch`. Overrides the three batched sub-process atoms (BMJ12 in `float64` tensors; kinematics as no-grad `[N]` tensors, CFF-bilinear/linear layers in-graph) + `setupKinematicsTorchBatch` (φ-independent BMJ12 quantities, 72 angular coeffs, one batched NN forward caching the CFF tensors). Unpolarized target only.

  It is a **safe scalar drop-in**: it does *not* override the inherited scalar `CrossSectionBH/VCS/Interf`, so driving it through the PARTONS pipeline runs native full-coverage BMJ12 arithmetic. The two method families are fully disjoint — the scalar ones read the base's **private** BMJ12 doubles (populated by `setKinematics`/`initModule`), the tensor ones read our `m_*Batch` members (populated by `setupKinematicsTorchBatch`). That private-member wall is *why* the BMJ12 formulae had to be re-transcribed rather than reused, and it's also why `m_preparedBatch` and the whole prepare/assemble split are invisible to the scalar path. Only `m_M` (proton mass, a constant) is shared unsuffixed between the two.

**`MathIntegratorModuleTorch`** (`Modules/`) — tensor twin of `MathIntegratorModule`. `integrateTorch()` evaluates a batched `[N]`-φ integrand and returns the integral in-graph. Supports the **same DEXP** integrator the scalar `DVCSAluMinusSin1Phi` uses (plus TRAPEZOIDAL/GL); `GK21_ADAPTIVE` is unsupported (non-constant-weight extrapolation). The fixed-rule (GL/TRAPEZOIDAL) reference nodes/weights are **cached** as `mutable` member tensors (`m_quadNodes`/`m_quadWeights`) — converted from NumA once and reused across calls, rebuilt only if the node count changes (and cleared by `setIntegrator` on a rule change); per call only the `[a,b]` remap runs (2026-06-29). DEXP has its own program-wide static table (`dExpTables()`, regenerated from the tanh-sinh closed form once at first use); TRAPEZOIDALLOG points are `[a,b]`-dependent and recomputed per call.

**Chain correspondence** (all base-typed pointers + virtual dispatch, same as scalar):

```
ObservableServiceTorch::computeManyKinematicTorch      ↔  ObservableService::computeManyKinematic
   computeSingleKinematicTorch                         ↔     computeSingleKinematic
ObservableTorch::computeTensorBatch (template method)  ↔  Observable::compute
   computeTensorImplBatch (hook)                       ↔     computeObservable
DVCSAluMinusTorch::aLUTensorBatch (pointwise)          ↔  DVCSAluMinus::computeObservable
DVCSProcessModuleTorch::crossSectionTensorBatch (Σ)    ↔  DVCSProcessModule::compute(…,VCSSubProcessType)
   crossSectionBH/VCS/InterfTensorBatch                ↔     CrossSectionBH/VCS/Interf
   setupKinematicsTorchBatch                           ↔     setKinematics + CFF forward
DVCSCFFModuleTorch::computeAllCFFsTensorBatch          ↔  DVCSConvolCoeffFunctionModule::computeCFF
```

`computeTensor`/`computeTensorImpl` (single-kinematic) survive as thin N=1 wrappers over their `…Batch` siblings — they are an entry-point convenience, not a separate implementation.

The only irreducible differences: the torch chain returns a grad-carrying `torch::Tensor` (scalar returns a detached `double` via the result bean), and evaluates the BMJ12 formulae + φ-nodes as `float64` tensors/batches (scalar uses native `double` + a scalar φ loop). With the same DEXP and `float64`, all three `observ_calc*` paths agree to every printed digit.

### ⚠️ Coverage caveat — torch path is unpolarized-target only

The torch BMJ12 port (`DVCSProcessBMJ12Torch`) implements **only the unpolarized-target sector** (Λ=0): only the unpolarized BH Fourier coeffs and the S=0 interference angular blocks (`m_Cang`/`m_Sang`) are transcribed; the LP/TP (`dC`/`dS`) rows are omitted. This is valid for A_LU (the only observable ported), whose target is unpolarized.

The hazard is **which scalar physics actually runs depends on which classes you wire**, because the torch leaves' scalar virtuals wrap the tensor methods:

- **Base PARTONS classes** (`DVCSProcessBMJ12` + `DVCSAluMinusSin1Phi`, as in `observ_calc()`) → PARTONS' native arithmetic → **full coverage** (polarized targets included). Use these for any polarized-target observable.
- **Torch leaves driven through the scalar service** (`observ_calc_torch_scalar()`): `DVCSAluMinusTorch::computeObservable` wraps `computeTensor().item()`, so the call routes into the **torch** chain (`crossSectionTensorBatch`), i.e. unpolarized-only physics — *not* the inherited native `CrossSectionBH/VCS/Interf`.

**The determining factor is always the observable leaf, not the process module:**

| Observable leaf | Process module | Physics that runs | Coverage |
|---|---|---|---|
| `DVCSAluMinusSin1Phi` (base) | `DVCSProcessBMJ12` (base) | native scalar | full |
| `DVCSAluMinusSin1Phi` (base) | `DVCSProcessBMJ12Torch` | native scalar | full |
| `DVCSAluMinusSin1PhiTorch` | `DVCSProcessBMJ12Torch` | tensor (unpolarized) | A_LU-type only |

Put a `*Torch` observable on top and you are in the tensor chain however you drive it; put a base observable on top and the torch process module behaves exactly like its base class.

So: **do not wire a `*Torch` observable/process leaf for a polarized-target observable and expect correct results** — its scalar path silently runs the unpolarized torch port (wrong/zero for the polarized contributions), with no error raised. For polarized-target work, use the base PARTONS classes (which remain registered alongside the torch ones). The torch leaves are correct only for unpolarized-target observables (A_LU and siblings). Removing this caveat requires porting the LP/TP coefficient rows into `setupKinematicsTorchBatch` + the sub-process tensor methods.

(Note: `DVCSProcessBMJ12Torch` only overrides the *tensor* sub-process methods `crossSectionBH/VCS/InterfTensor`; it inherits PARTONS' scalar `CrossSectionBH/VCS/Interf` unchanged. So a torch *process* wired under a *base* `DVCSAluMinusSin1Phi` observable would still run full-coverage native scalar physics — the unpolarized restriction only bites along the tensor path, reached via a torch *observable* leaf.)

## Data format

Input CSVs are pipe-separated (`|`), **observable format**: `xB | t | Q2 | E | phi | <observable> | error`. `CFF_NN_Fitter::load_data_observable()` returns `(X[N,3]=(xB,t,Q2), E[N], phi[N], y_obs[N]=col 5, sigma[N]=last col)`. Training and prediction operate on the observable (A_LU^{sin1φ}). φ is loaded and passed into the kinematics — the sin1φ moment integrates it out, but it's kept so `CustomLoss` is reusable for φ-dependent observables.

The old CFF-label loader `CFF_NN_Fitter::load_data()` (which read `…|ImH|ReH|…` columns as NN targets) was **removed** (2026-06-18) — the workflow now fits the observable, not CFF labels. (`NN_Fitter::load_data()` in the separate `NN_Fit.{h,cpp}` is unrelated and still used by the `NN_CFF_fit` executable.)

Data path and output paths are hardcoded absolute paths in `src/Run_CFF_NN_Fit.cpp` and `src/NNFit/CFF_NN_Fit.cpp` (pointing to `My_Analysis/Partons_output/`). Update these when moving environments. Current data file: `Data/Partons_input/BSA_CLAS_07_KK_format_ALU_error.csv` (16 points).

## Output files (`My_Analysis/Partons_output/`)

| File | Content | Written by |
|---|---|---|
| `cff_learning_curve.csv` | epoch, train reduced χ²/n, val reduced χ²/n (every 2 epochs) — central fit | `train_nn()` (via `fit_once()`) |
| `cff_learning_curve_last_replica.csv` | same format, for the **last** replica only (`r == n_replicas-1`); other replicas pass `""` to skip. A replica-fit diagnostic — it is fit to smeared pseudodata, so do **not** compare its χ²/n to the central fit's against real data. `fit_once()` truncates on open, so across retries the surviving file is the kept attempt's curve. | `train_replicas()` (via `fit_once()`) |
| `obs_prediction.csv` | `xB,t,Q2,E,phi,obs_true,obs_pred,error` per point | `predict()` |
| `obs_model_eval.csv` | `observable,mse,r_squared,chi2` (chi2 = reduced χ²/n) | `predict()` |
| `cff_model.json` | trained NN export — `arch`, `dtype`, `best_val_chi2`, `input_features`, `x_pow`, `output_layer` (= `m_output_layer`), min-max `scaling`, and `fc1`/`fc2` weights+biases. Lets the exact NN forward be reproduced out-of-process (CFF scans/plots in `CFF_obs_train_predict_plot.ipynb`) | `predict()` (`export_model_json`) |
| `cff_model_replica_<NN>.json` | same format as `cff_model.json`, one per trained replica (`<NN>` = zero-padded replica index) — for Python mean ± σ CFF bands. The whole set is deleted and rewritten on each export, so the directory never mixes runs | `export_replicas()` |

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

---

## Session notes (2026-06-18)

### Goal

Implement `CustomLoss` for real and make the whole `CFF_NN_Fitter` workflow fit the **observable** (A_LU^{sin1φ}) instead of CFF labels. (This supersedes the *aspirational* `CustomLoss` description in the 2026-06-02 notes, which differed from what was actually built — see "Divergences" below.)

### What was implemented

**`CustomLoss`** (`include/NNFit/CustomLoss.h`, `src/NNFit/CustomLoss.cpp`) — `CustomLossImpl : torch::nn::Module`, wrapped `TORCH_MODULE(CustomLoss)` so it's instantiated/called like a PyTorch loss.
- **Constructor** `(CFFNNModel net, outputLayer, xMin={}, xMax={})` builds and wires the `*Torch` chain **once** (same wiring as `observ_calc_torch()`): `DVCSCFFNNTorch` (← `setModel(net, …, xMin, xMax)`) + xi-converter + scales + `DVCSProcessBMJ12Torch` + `DVCSAluMinusSin1PhiTorch`; fetches `DVCSObservableServiceTorch` by name; cross-casts the observable to `DVCSObservableTorch*`. The `net` is shared by handle, so optimizer updates are seen by the chain.
- **`forward(X[N,3], E[N], phi[N], y_obs[N], sigma[N])`** loops rows, builds `DVCSObservableKinematic(xB,t,Q2,E,phi)`, calls `m_pServiceTorch->computeSingleKinematicTorch(kin, m_pObsTorch)` (grad-preserving), and accumulates `χ² = Σ ((pred − y_obs)/sigma)²` into a 0-d `float64` tensor connected to the NN parameters.
- Drives through `DVCSObservableServiceTorch` (not `computeTensor` directly) — i.e. the planned 2026-06-16 "decided" item is done.

**`load_data_observable()`** — reads the observable CSV `xB|t|Q2|E|phi|<obs>|error` and returns the **5-tuple** `(X[N,3], E[N], phi[N], y_obs[N]=col 5, sigma[N]=last col)`. **φ is kept** (passed into the kinematic) so the loss is reusable for φ-dependent observables.

**`train_nn()` rewired** — uses `load_data_observable()`, splits all 5 tensors, fits `m_X_min`/`m_X_max` on the **raw** training kinematics (passed to `setModel`; the NN module scales internally, so `X` stays raw because the chain builds kinematics from raw values), constructs `CustomLoss(net, m_output_layer, m_X_min, m_X_max)` once, and per epoch does `loss = loss_fn(X_train, E_train, phi_train, y_train, s_train); loss.backward(); optimizer.step()`. Early-stopping/learning-curve unchanged.

**`predict()` rewritten** — evaluates the **observable** through the same `*Torch` chain per point (`computeSingleKinematicTorch`, `NoGrad`) and compares to the measured value; writes `obs_prediction.csv` + `obs_model_eval.csv` (MSE, R², χ²). Replaces the old CFF-label prediction.

**`CFF_NN_Fitter::load_data()` removed** — the CFF-label loader had no remaining callers after the rewrites (dead code). Declaration + definition deleted.

**Data file** switched to `BSA_CLAS_07_KK_format_ALU_error.csv` (16 points, vs 10). **`CMakeLists.txt`** adds `src/NNFit/CustomLoss.cpp` to `Run_CFF_NN_Fit` (+ `REMOVE_ITEM` from the glob target).

### Divergences from the 2026-06-02 (aspirational) note

The earlier note predicted a different `CustomLoss` than what was built. Actual implementation: φ is **kept** (5-tuple loader, not dropped/4-tuple); min-max scaling **is** applied (passed via `setModel`, not "removed/identity"); `load_data()` was **removed** (not "alongside"); and the loss **drives through `DVCSObservableServiceTorch`** (not `computeTensor` directly).

### Verification

Builds clean. Training runs on the 16-point file with **χ² decreasing monotonically** (confirms the autograd path NN→CFFs→A_LU→χ² is intact). All three `observ_calc*` paths still agree to every printed digit; `observ_calc_torch()` keeps `requires_grad=true`.

### Known performance characteristics & planned speedups (branch `5-speedup-…`)

Per epoch = N rows × (DEXP `L` levels × 2 helicities) `crossSectionTensor` calls, each self-contained (re-runs `setupKinematicsTorch` → NN forward + 72 coeffs + Fourier contractions). φ is batched within a call; refinement levels, helicities, and data points are **not** batched. Planned, independent optimizations:
- **#1 hoist setup** — compute φ-independent setup + Fourier coeffs once/point (add a lightweight `crossSectionTensor(λ,charge,φ)` overload that assumes a `prepare(kin)` ran): setups `2L → 1`/pt.
- **#2 fixed quadrature** — swap DEXP → fixed GL/trapezoidal in `DVCSAluMinusSin1PhiTorch` (verify it matches DEXP): integrand evals `L → 1` (cross-section calls `2L → 2`). Per-observable choice; validate per integrand.
- **#3 batch across data points** — a new `computeAluBatch(X[N,3],E[N])` theory entry (kinematics as `[N]`, one `[N,3]` NN forward, σ as `[N,M]`); bypasses the per-kinematic PARTONS API, coexists with the single-kinematic chain. For large N.
- Optional: multithread the per-row loop with `std::thread`/`std::async` over a **clone-per-thread** chain pool (share the NN, `torch::set_num_threads(1)`); do **not** reuse `computeManyKinematic` (bean pipeline, detached). Worth it only for large N, after #1.

---

## Session notes (2026-06-22)

### Speedup options — design discussion (no code yet)

Detailed analysis of the four speedup options for the per-epoch χ² training cost. **Decision: implement in order #1 → #2 → #3** for a growing-N dataset. #3 and #4 are **alternatives, not complements** (both target the N-data-points axis; after #3 there is no per-point loop left for #4 to thread). #3 subsumes #1's benefit and requires #2.

**The four options:**

| # | Name | What it parallelizes / removes | Effect | Risk |
|---|---|---|---|---|
| **#1** | Hoist setup (prepare/assemble split) | repetition *within* one kinematic point — split φ-independent **prepare** (NN forward + kinematics + 72 coeffs) from φ-dependent **assembly** | setups/point `2L → 1` | low (pure refactor) |
| **#2** | Fixed quadrature | adaptive DEXP → fixed GL/trapezoid in `DVCSAluMinusSin1PhiTorch` | integrand evals `L → 1`; gives a **static** φ grid | low–med (validate vs DEXP per integrand) |
| **#3** | Batch across data points (vectorization) | the per-point loop — kinematics as `[N]`, one `[N,3]` NN forward, σ as `[N,M]`, φ-integral over N axis; + `computeManyKinematicTorch` service driver | removes per-point loop; multicore **free** via ATen intra-op; GPU-ready; scales past core count | medium (invasive but deterministic, testable) |
| **#4** | Parallel per-point (multithreaded) | whole per-point computations across a **clone-per-thread** pool; share `CFFNNModel`, parallelize forward, **one** `backward()` on main thread; `set_num_threads(1)` | speedup ≤ core count, minus thread/clone/join overhead | high (concurrency; shared `.grad`/caches) |

**Key conclusions from the discussion:**

- **#1 before #3** — #1 is the exact prepare/assemble factoring #3 needs (#3 = "lift that split to `[N]` tensors"); doing it first on the simpler single-point path de-risks #3 and is a standalone win for today's training + the `observ_calc_torch*` verification calls. Verification gate: `observ_calc_torch()` must stay equal to `observ_calc()` to every digit.
- **#2 is a prerequisite for #3** — an `[N,M]` *static* grid cannot hold DEXP's per-point adaptive refinement level, so #3 drags in the fixed-quadrature swap.
- **#3 vs #4 — why #3 is safer & cleaner as N grows:** #4 is **task parallelism** (you spawn threads; whole per-point computations share mutable state — module member caches *and* the parameter `.grad` buffers → write–write races; safety is your burden, nondeterministic bugs if wrong). #3 is **data parallelism / vectorization** (zero threads in your code; parallelism lives *inside* each tensor op on **disjoint** output elements, guaranteed by ATen/BLAS; graph built once, **one** deterministic `backward()` — safe by construction). Both light up all cores; only #4 owns the locking.
- **Why PARTONS' `computeManyKinematic` is safe-parallel but its pattern can't be borrowed for training:** PARTONS gets safety from (a) **clone-per-thread** (transfers to #4) + (b) **detached independent scalar outputs, no backward, no shared `.grad`** (does *not* transfer — training needs a shared graph + shared gradient accumulation). The detachment that makes it thread-safe is exactly what severs the gradient → unusable for training. Hence "do not reuse `computeManyKinematic`."

### #3 design decisions (so a new observable does *not* reimplement the whole batched stack)

- **Layered, link-for-link — not a monolithic `computeAluBatch`.** Push the batch dimension **down** into the reusable layers (batched CFF forward, batched `crossSectionTensorBatch`, batched `integrateTorch` over N). Then adding a new observable means writing **only its thin `computeTensorImplBatch` leaf** (combine the already-`[N,M]` cross-sections + apply its Fourier weight) — same "add a leaf, reuse everything below" property the scalar/single-kinematic chains already have. The flat `computeAluBatch` framing in the 2026-06-18 note is the anti-pattern (forces per-observable reimplementation).
- **Add batched methods to the *existing* modules** (third sibling API alongside the scalar virtual + single-kinematic tensor method) — **not** new modules. Reuses the configured/registered chain, the same `CFFNNModel` handle, and keeps the formulae next to their twins. The only new *type* is a small stateless struct carrying the `[N]` φ-independent setup + cached CFF tensors (like `AllCFFsTensor` / `Theory::VCSCoeffs`/`InterfCoeffs`), no registration. Add the batched virtual to the generic templates (`ObservableTorch<K>`, `ProcessModuleTorch<K>`) too, for channel symmetry.
- **Service: add `computeManyKinematicTorch`, leave `computeSingleKinematicTorch` untouched.** Mirror the scalar `computeSingleKinematic ↔ computeManyKinematic` pair. Thin driver in the generic `ObservableServiceTorch<K>` mixin: takes a `DVCSObservableKinematicList` (PARTONS-symmetric) → `obs->computeTensorBatch(...)` → `[N]` grad-carrying tensor. Channel-specific unpacking (list → `xB[N],t[N],Q²[N],E[N]`) lives in the process leaf's `prepareBatch`, not the service. Optional raw-tensor overload `(X[N,3],E[N],phi[N])` for the hot training loop to skip bean construction.
- **Formulae are reused, not re-derived.** Tensor-algebra layers (CFF bilinear/linear, cross-section assembly) are **shape-polymorphic** — broadcast from `[Nφ]` to `[N,M]` with the same expressions. Only the kinematics block (currently "verbatim doubles, no grad") must be **re-expressed** as `[N]`-tensor arithmetic (same formula, double→tensor). Write it **shape-polymorphic** so the single-point path calls it with N=1 — collapsing the single-point-tensor and batched-tensor copies into one; only the PARTONS scalar-`double` virtual stays separate (required by the scalar drop-in contract).
- **Scalar-chain compatibility preserved by construction:** #3 is purely additive (new entry points bypass the single-point PARTONS API); the scalar virtuals and the single-kinematic tensor chain are never modified.

### Verification gates (per step)

| Step | Gate |
|---|---|
| #1 | `observ_calc_torch()` unchanged vs `observ_calc()` to all printed digits |
| #2 | fixed-rule φ-integral matches DEXP to tolerance (per integrand) |
| #3 | N=1 batch == single-kinematic result; full-batch χ² == per-point-loop χ² |

### Side observation (from the fit run reviewed this session)

The poor fit (`R²=−37`, χ²≈1798, predictions 5–50× too small) was **undertraining, not a chain bug**: `max_epochs=100`/`patience=20` in `train_nn()` (vs the 10000/200 the codebase intends) stopped training while val loss was still decreasing monotonically (~3% drop over 100 epochs, no plateau, early stopping never triggered). All three `observ_calc*` paths still agreed to every digit. Fix: restore `max_epochs`/`patience` (and optionally raise lr `1e-4 → 1e-3`).

### #2 implemented (DEXP → fixed GL-10) and verified

Branch `5-speedup-singlecalcphiindependentkin-GLint`. `DVCSAluMinusSin1PhiTorch`'s constructor now selects a fixed 10-point Gauss–Legendre rule instead of DEXP for the φ-integral (`DVCSAluMinusSin1PhiTorch.cpp:21`):

```cpp
// was: MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::DEXP);
MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 10);
```

Rationale: the `A_LU^{sin1φ}` integrand is smooth and 2π-periodic, so a fixed rule is **one batched integrand evaluation** vs DEXP's adaptive multi-level (`L+1` evaluations) — the #2 speedup. Only this leaf changed; `MathIntegratorModuleTorch` already supported GL.

**Verification gate passed.** Ran `Run_CFF_NN_Fit` and compared the three `observ_calc*` paths at xB=0.2, t=−0.2, Q²=2, E=5.932 (same NN weights; a deliberately short 5-epoch run — fit quality irrelevant, only cross-path agreement matters):

| Path | Quadrature | A_LU^{sin1φ} |
|---|---|---|
| `observ_calc()` (base PARTONS, scalar) | DEXP adaptive | **−0.00895582** |
| `observ_calc_torch()` (torch tensor path) | **GL-10** | **−0.00895585** |
| `observ_calc_torch_scalar()` (torch scalar virtuals) | GL-10 | −0.00895585 |

GL-10 vs scalar DEXP differ by **~3×10⁻⁸ absolute (~3.4×10⁻⁶ relative) — agreement to ~6 sig figs**, the same tiny GL-vs-adaptive gap seen historically, far below fit/measurement precision. So #2 reproduces the DEXP value for this integrand while collapsing the φ-integral to one batched eval.

**Caveats:** validated at a single kinematic point — worth a spot-check at extreme kinematics (low/high xB, large |t|) if the dataset spans a wide range (almost certainly fine for a smooth periodic integrand). The change is per-observable: re-validate GL vs DEXP for any new integrand before swapping. The #2 edit is currently uncommitted on the branch.

**Run invocation note** (rediscovered this session): `Partons::init` derives the properties-file dir from `argv[0]` (`Partons.cpp:81–88`), but the paths *inside* `bin/partons.properties` (`bin/logger.properties`, `data/xmlSchema.xsd`) are relative to the **actual CWD**. So run from the `DVCS_analysis/` dir as `./bin/Run_CFF_NN_Fit` (not `cd bin && ./Run_CFF_NN_Fit`, which makes the inner `bin/…` paths resolve to `bin/bin/…` and fails on `logger.properties`).

---

## Session notes (2026-06-24)

### #1 implemented (hoist setup — prepare/assemble split) and verified

Branch `6-speedup_via_vectorized_batchobscalc`. Removed the redundant per-helicity recomputation of the φ-independent setup in the torch cross-section path. Previously `DVCSAluMinusTorch::aLUTensor` called the self-contained `crossSectionTensor(λ, charge, **kin**, φ)` **twice** (λ=±1), and that overload ran `setupKinematicsTorch` at the top of each call — so the helicity-independent work (one NN forward + the verbatim BMJ12 kinematic block + 72 angular coeffs) executed **twice per data point**.

**Why it was redundant:** beam helicity enters only the φ-dependent assembly; the CFFs and BMJ12 kinematics do not depend on λ. (Confirmed the scalar path has the *same* structure — `DVCSAluMinus::computeObservable` also calls `compute(+1,…)`/`compute(−1,…)` and each `DVCSProcessModule::compute` re-runs `setKinematics` + conditional `computeConvolCoeffFunction` — but scalar dodges the cost via `m_isCCFModuleDependent` gating + the CCF module's by-kinematics cache. The torch path can't use a detached cache without severing the autograd graph, so it must hoist instead.)

**What changed (2 source files):**

- **`DVCSProcessModuleTorch.h`** (channel-layer base; header-only) — split the monolithic `crossSectionTensor` into:
  - public **`prepareTensor(kin)`** → calls `setupKinematicsTorch(kin)` once, sets `m_prepared = true`;
  - **lightweight** `crossSectionTensor(λ, charge, φ)` and `crossSectionTensor(λ, charge, φ, processType)` — assemble-only (the former VCS/BH/INT summation body), guarded by `m_prepared` (throws `ElemUtils::CustomException` if called before prepare);
  - **self-contained** `crossSectionTensor(λ, charge, kin, φ[, processType])` — kept for single-shot callers, now delegate: `prepareTensor(kin)` → lightweight assemble;
  - new protected member `bool m_prepared = false`.
  Overloads disambiguate by arg list (the kin-taking ones have an extra `DVCSObservableKinematic&`; no ambiguity vs the `phi`-only ones).
- **`DVCSAluMinusTorch.cpp`** (`aLUTensor`) — now `pProc->prepareTensor(kin)` once, then lightweight `crossSectionTensor(+1.,−1.,φ)` / `crossSectionTensor(−1.,−1.,φ)`.

The generic `ProcessModuleTorch<K>` was **not** touched — it intentionally carries no cross-section/setup API (channel-specific by design), so the split lives in the DVCS channel layer. This prepare/assemble factoring is also exactly the split #3 will lift to `[N]` tensors, so #1 de-risks the batched work on this branch.

**Verification gate passed.** Built clean (`make Run_CFF_NN_Fit`); only `aLUTensor` uses the new lightweight overload. Ran `Run_CFF_NN_Fit` (fresh-seeded NN — only cross-path agreement matters):

| Path | A_LU^{sin1φ} | grad |
|---|---|---|
| `observ_calc()` (base PARTONS, scalar) | **0.089754** | — |
| `observ_calc_torch()` (torch tensor path) | **0.0897542** | `requires_grad = true` |
| `observ_calc_torch_scalar()` (torch scalar virtuals) | 0.0897542 | — |

All three agree to every printed digit (the 6th-digit spread is the unchanged GL-10-vs-DEXP gap, not a #1 regression); the tensor path keeps its gradient; training χ² fell monotonically (1594→159), confirming the NN→CFFs→A_LU→χ² autograd path survives the hoist.

**Note on the gradient:** before, two NN forwards built two CFF subgraphs (same values); after, one forward's CFF tensors feed both σ⁺ and σ⁻ — the value and gradient are identical (chain rule through the shared node), and it's more faithful (the CFFs at a point *are* the same object for both helicities).

**Per-point setup cost:** NN forward + 72-coeff build + BMJ12 kinematic block now run **once** per data point instead of twice. Uncommitted on the branch.

---

## Session notes (2026-06-29)

Branch `adding_replicas` (the #1 hoist-setup + CFF-export changes carried over here; all uncommitted).

### `predict()` now exports the trained model — `cff_model.json`

`CFF_NN_Fitter::predict()` was reworked around an out-of-process CFF workflow:

- **Added `export_model_json(path)`** (private, called at the end of `predict()`) — writes `My_Analysis/Partons_output/cff_model.json`: `arch`, `dtype: float32`, `input_features`, `output_layer` (= `m_output_layer`), min-max `scaling` (`x_min`/`x_max`), and `fc1`/`fc2` weights+biases. Weights are written in PyTorch's `[out, in]` orientation (the forward is `y = x Wᵀ + b`), at `setprecision(9)` to round-trip float32. Lets the exact NN forward be reproduced in Python (numpy: `tanh((x−xmin)/(xmax−xmin) @ W1ᵀ + b1) @ W2ᵀ + b2`) with **no torch dependency** — chosen over `.pt` (a hand-written C++ `nn::Module` doesn't cross cleanly to Python except via TorchScript; the net is tiny so the numpy forward is trivial and was validated to ~5e-6).
- **Removed the interim `cff_model_eval.csv` block** (the per-point NN CFF dump) — it had served its purpose validating the Python forward against the C++ outputs; that validation is done, so the writer (and the now-orphaned notebook validation cell) were deleted.
- **Notebook `CFF_obs_train_predict_plot.ipynb`** gained a CFF-scan section: loads `cff_model.json`, reproduces the forward, and plots **CFFs vs xB** (fixed t, Q²) and **CFFs vs −t** (fixed xB, Q²) with the fixed kinematics annotated and `savefig` to PNG. One line per CFF in `output_layer` (just `ImH` in the current config).

### GL/TRAPEZOIDAL node caching in `MathIntegratorModuleTorch`

`integrateTorchQuadrature` previously re-created the `[M]` node/weight tensors (`torch::tensor(refNodes,…)`) on **every** `integrateTorch` call. Now they're cached in `mutable m_quadNodes`/`m_quadWeights`, built once and reused; rebuilt only if the cache is empty or the node count changed, and cleared by `setIntegrator` on a rule change (covers GL→TRAPEZOIDAL at the same N). Per call only the `[a,b]` remap `x = d + c·m_quadNodes` runs. Pure caching optimization — **value-preserving** (verification gate below passed). For GL-10 the saving is tiny; it matters more at high node counts / in a hot batched loop. (DEXP already used its static `dExpTables()`; TRAPEZOIDALLOG is inherently per-`[a,b]`.)

**Verification (caching):** `Run_CFF_NN_Fit` at xB=0.2, t=−0.2, Q²=2, E=5.932 — `observ_calc()` = **0.124825**, `observ_calc_torch()` = **0.124826** (`requires_grad=true`), `observ_calc_torch_scalar()` = 0.124826. All agree to printed digits (the 6th-digit gap is the unchanged GL-vs-DEXP-scalar difference, not a regression).

### Design discussion (no code): moving CFF fitting to Python

Decided the path for a future Python-driven fit: **wrap the validated C++ torch chain via pybind11** (the PARTONS `python_interface` branch) rather than reimplement the physics or adopt Gepard. Key facts established: C++ libtorch and Python torch share one ATen/autograd engine, so a `torch::Tensor` returned from C++ *is* a `torch.Tensor` in Python with the graph intact (`backward()` in Python reaches the params) — **provided the extension links the exact libtorch Python ships** (ABI/cxx11-ABI match is the make-or-break risk). Chosen seam: **at the CFF level** — NN + training in Python, C++ wraps `CFFs + kinematics → A_LU` (add a `prepareTensorWithCFFs` that injects CFFs instead of the C++ NN forward). The wrapper adds negligible compute overhead (zero-copy tensor passing, µs-scale pybind crossing) **as long as the boundary is coarse** (one batched `predict` per epoch, not per point).

### Replica ensemble plan (no code)

For CFF uncertainty bands: train N replicas, save each as `cff_model_<r>.json`, read+average in Python (mean ± σ via `fill_between`). The σ is only meaningful with **Monte Carlo data replicas** — fluctuate each point `yᵢ → yᵢ + σᵢ·zᵢ` per replica (σ kept for the χ² weighting), seeded per replica for reproducibility — **not** seed-only reruns (that's optimization scatter, demonstrated by three runs swinging R² −0.20 → 0.64 → 0.005). Data fluctuation is a training-code change needed regardless of orchestration (C++ loop vs shell `--replica r`); 10 replicas is few (σ noisy), 100+ preferred.

---

## Session notes (2026-08-05)

### `CFF = xB^x_pow * NNet_output` prefactor

Branch `Imp_replica_fits`. Added an optional power-law prefactor so the NN can learn a rescaled target instead of the raw CFF directly — motivated by CFFs (e.g. ImH) that vary over orders of magnitude near small xB, where a plain NN output is harder to fit than `xB^p · NN(x)` for a suitable `p`.

**What changed:**

- **`DVCSCFFNNTorch`** — new `double m_xPow` member (default `0.0`); `setModel(net, outputLayer, xMin, xMax, xPow = 0.0)` gained the parameter; `forwardNN()` now returns `NNet_output * xB^m_xPow` (via `std::pow`) instead of the raw forward output. `xPow = 0.0` is a no-op (`xB^0 = 1`), so this is backward compatible with every existing call site that doesn't pass it.
- **`CFF_NN_Fitter`** — new constructor parameter `double x_pow = 0.0`, stored as `m_xPow` and threaded through every `setModel(...)` call site (`train_nn()` via `CustomLoss`, `predict()`, `observ_calc()`, `observ_calc_torch()`, `observ_calc_torch_scalar()`) so training and every inference/verification path apply the identical rescaling. Also written into `cff_model.json` as `"x_pow"` (`export_model_json()`), so the Python-side CFF-scan notebook can reproduce the exact forward.
- **`CustomLossImpl`** — constructor gained `double xPow = 0.0`, forwarded straight to `setModel`.
- **`Run_CFF_NN_Fit.cpp`** — `CFF_NN_Fitter` construction passes `x_pow = 0.0` explicitly (commented: "NN learns xB*CFF (~ O(1) near small xB)" for the `x_pow = 1.0` use case), i.e. the feature is wired but off by default in this run.
- **Also in this commit**: removed `weight_decay(1e-3)` from the Adam optimizer in `train_nn()` (regularization was suppressing fit quality on the small 16-point dataset; no replacement regularization added).
- New notebook `CFF_obs_train_predict_plot_xpow.ipynb` (xB-scan plots for the rescaled-CFF case).

**Not yet done at this point**: `x_pow` is a manually-set constant per run, not fit or selected automatically; no systematic comparison of `x_pow` values was recorded in-session.

---

## Session notes (2026-09-01)

### Goal

Two goals, closing issue #12: (1) put the training loss on a scale that's comparable across dataset/split sizes and reads as a standard reduced-χ² diagnostic; (2) implement the Monte Carlo replica ensemble planned in the 2026-06-29 notes, following Gepard's `datasets_replica_vectloss` / `train_net_vectloss` conventions (see also `[[replica-smearing-gepard-method]]` in memory).

### Loss switched to reduced χ²/n

**`CustomLossImpl`** — constructor and `forward` gained `bool normalize = true`; `forward` now returns `chi2 / n` when `normalize` is true (the default) instead of the raw `Σ ((pred−y)/σ)²` sum. `normalize = false` is kept only for controlled A/B comparisons against the reduced-χ² default (matches Gepard's `CustomLoss`/`CustomLoss_vectloss`, which use `torch.mean` rather than a raw sum). `predict()`'s reported `chi2` in `obs_model_eval.csv` switched from `.sum()` to `.mean()` for the same reason — it and `cff_model.json`'s `best_val_chi2` are now on the same reduced-χ²/n scale as the training loss.

### `train_nn()` refactored around a shared `fit_once()` helper

`train_nn()`'s body (shuffle/split, per-feature min-max on the training split, fresh net+optimizer, the epoch loop with early stopping and best-val snapshotting) was extracted into a private helper:

```cpp
struct TrainedModel { CFFNNModel net; torch::Tensor X_min, X_max; float best_val_loss; };
struct FitOutcome    { TrainedModel model; bool hopeless; };
FitOutcome fit_once(X, E, phi, y_obs, sigma, bool smear,
        const std::string& learning_curve_path, float hopeless_val_loss,
        int hopeless_check_epoch, unsigned seed, bool normalize_loss = true) const;
```

`fit_once` is a pure function of its arguments (no member state touched except via the returned `TrainedModel`) — this is what let the same routine serve both the central fit and every replica. Key behaviors:

- `torch::manual_seed(seed)` fixes both weight init and the smear draw; a separate `std::mt19937 rng(seed)` (same seed) fixes the train/val shuffle — so an identical `seed` reproduces byte-identical starting conditions (used for the normalize-loss A/B check below).
- `smear = true` trains on `y_used = y_obs + N(0, sigma) * sigma`-scaled noise (`y_obs + torch::randn_like(y_obs) * sigma`) instead of raw `y_obs`; `sigma` itself is never smeared (it stays the χ² weight). `smear = false` (the central-fit case) trains on raw `y_obs`.
- **Hopeless-attempt detection**: aborts early (`hopeless = true`, `FitOutcome` returned immediately) if val loss is ever non-finite (NaN/Inf), or if it's still above `hopeless_val_loss` at an epoch that's a multiple of `hopeless_check_epoch` (periodic re-check, matching Gepard's `train_net_vectloss` re-checking every validation batch rather than a single fixed checkpoint). `hopeless_check_epoch <= 0` disables the periodic check entirely — used by `train_nn()`'s central fit, which keeps its original "always run to completion or early-stop" behavior.
- `learning_curve_path` may be `""` to skip writing a CSV (used by every replica; only the central fit writes `cff_learning_curve.csv`).
- `restore_best()` now guards against an empty snapshot (no improving epoch yet before a hopeless abort) by leaving the freshly-initialized weights in place rather than crashing.

`train_nn()` itself is now a thin wrapper: loads data, calls `fit_once(..., smear=false, "cff_learning_curve.csv", max_float, hopeless_check_epoch=0, seed=random_device{}())`, and copies the returned `TrainedModel` into `m_net`/`m_X_min`/`m_X_max`/`m_best_val_loss`.

### `train_replicas()` and `export_replicas()` (new)

```cpp
void train_replicas(int n_replicas = 10, int max_retries_per_replica = 5,
        float hopeless_val_loss = 100.f, int hopeless_check_epoch = 200,
        unsigned base_seed = 0, bool normalize_loss = true);
void export_replicas(const std::string& out_dir,
        const std::string& name_prefix = "cff_model_replica_") const;
```

- `train_replicas` loads the data once, then loops `r = 0..n_replicas-1`; for each replica it retries `fit_once(..., smear=true, ...)` up to `max_retries_per_replica` times, redrawing everything (fresh smear + fresh split + fresh weights, not just a weight reinit) whenever an attempt comes back `hopeless`. If still hopeless after all retries, the last attempt is kept anyway with a logged warning (so a run always produces exactly `n_replicas` models). Results accumulate in `std::vector<TrainedModel> m_replicas`.
- **Seeding**: `base_seed = 0` (default) draws every attempt's seed from `std::random_device` — true per-attempt independence, the normal production mode. A nonzero `base_seed` makes seeds deterministic (`base_seed + r*100 + attempt`), reproducing byte-identical smear/split/init across two runs that should differ only in one deliberately-varied parameter — used to A/B `normalize_loss` (true vs false) under otherwise-identical starting conditions.
- `export_replicas` writes each `TrainedModel` via a new overload of `export_model_json(path, net, xMin, xMax, bestValLoss)` — the original `export_model_json(path)` (used by `predict()` for the central fit) is now a thin wrapper calling this with `m_net`/`m_X_min`/`m_X_max`/`m_best_val_loss`. Output filenames are `<name_prefix><NN>.json` with the replica index zero-padded to 2 digits (default prefix `cff_model_replica_`); `name_prefix` can be overridden for a distinct export set (e.g. an A/B comparison run).

### Early stopping retuned

`patience` raised `200 → 1000` (max_epochs unchanged at 10000) — with the loss now normalized (reduced χ²/n rather than a raw sum that scaled with dataset size), the loss landscape is flatter and needed more patience to avoid stopping on noise before convergence.

### `Run_CFF_NN_Fit.cpp` wiring

After the existing `train_nn()`/`predict()`/`observ_calc*()` calls: `fitter.train_replicas(10)` then `fitter.export_replicas(out_dir)`. Also added a `std::chrono::steady_clock` wall-time measurement around `main()`'s body, printed as `"Total run time: <s> s"` at exit (replica training multiplies the single-fit cost by `n_replicas × (1 + retries)`, so this is now worth tracking).

### Verification

Built clean; ran `Run_CFF_NN_Fit` end-to-end. Training loss reads as a reduced χ²/n (order-1 for a reasonable fit, rather than scaling with the 16-point dataset size). `train_replicas(10)` produced 10 accepted replicas (no exhausted retries in the observed run) and `export_replicas` wrote 10 `cff_model_replica_NN.json` files alongside the existing `cff_model.json`. All three `observ_calc*()` cross-path-agreement checks from prior sessions were not specifically re-verified this session (no BMJ12/torch-chain physics touched — only the loss normalization and the training/orchestration layer above it).

### Known limitations / open tasks

- `x_pow` (2026-08-05) is still a manually-set constant, not fit or selected automatically.
- `hopeless_val_loss = 100.f` / `hopeless_check_epoch = 200` / `max_retries_per_replica = 5` are initial defaults, not yet tuned against the actual observed replica-loss distribution on the 16-point dataset. *(The retry parameter was renamed `max_tries_per_replica` and its default raised to 30 on 2026-09-18, along with the exhaustion policy — see that session's notes. The threshold values still stand as written.)*
- Per-replica `fit_once` calls are still sequential (no threading); `n_replicas × (1+retries)` full training runs is the dominant cost noted in "Run_CFF_NN_Fit.cpp wiring" above — a candidate for a future speedup pass (independent per-replica RNG/graphs make this an easier parallelization target than the earlier #4 per-point-threading idea, since there's no shared-gradient race: each replica has its own `net`/optimizer end-to-end).
- `CFF_plots_ALU_2007_xpow_replica_Farm.ipynb` (untracked, in `My_Analysis/Codes/`) appears to be in-progress replica-band plotting work, not yet committed.
- **(2026-09-15) A future dataset is planned that trains on raw per-phi A_LU directly**, rather than the sin1φ Fourier moment used everywhere today (`DVCSAluMinusSin1PhiTorch`). When that work starts, `DVCSAluMinusTorch::computeTensorImplBatch(List<K>)` (branch `vect_optionA`) needs a real implementation instead of its current throwing placeholder — it needs each of the N kinematics paired with its *own* phi (an `[N]` own-phi broadcast), whereas the existing `aLUTensorBatch`/`crossSectionTensorBatch` machinery broadcasts phi as an `[M]` axis *shared* across all N points (an `[N,M]` outer product, correct for Gauss-Legendre quadrature over the sin1φ moment but not for pointwise data). Plug the new method into the same `...Batch`-suffixed chain (`aLUTensorBatch` → `prepareTensorBatch`/`crossSectionTensorBatch` → `setupKinematicsTorchBatch` → `computeAllCFFsTensorBatch`) that `DVCSAluMinusSin1PhiTorch::computeTensorImplBatch` already uses for the moment case.

---

## Session notes (2026-09-16)

Branch `vect_optionA`. Commit `82edefd` — **deleted the dead single-point torch path** (6 files, +69/−902). The "Theory submodule" and coverage-caveat sections above were rewritten to match; earlier dated session notes were left alone as historical record, so they still describe the now-removed single-point API.

### Why it was dead

The single-kinematic tensor machinery lost its last caller in `893b5d8` (Option A), when `DVCSAluMinusSin1PhiTorch::computeTensorImpl` became a thin N=1 wrapper around `computeTensorImplBatch`. From that commit on, nothing drove `prepareTensor`/`crossSectionTensor`; the batched path at N=1 computes the same formulae. It survived only because the 2026-06-22 design note deliberately landed `#3` as **purely additive** (keeping the old path as a live reference during validation) — correct during the migration, just never cleaned up.

Confirmed dead before deleting: repo-wide grep found no caller outside the definitions themselves; every site that instantiates an observable (`CustomLoss.cpp:55`, `CFF_NN_Fit.cpp:368,602,669`) uses `DVCSAluMinusSin1PhiTorch::classId`, never bare `DVCSAluMinusTorch`; and there is no `tests/` directory.

### What was removed

| File | Removed |
|---|---|
| `DVCSProcessModuleTorch.h` | `prepareTensor`, all four non-batch `crossSectionTensor` overloads (2 self-contained + 2 lightweight), the three non-batch sub-process pure virtuals, `setupKinematicsTorch`, `m_prepared` |
| `DVCSProcessBMJ12Torch.{h,cpp}` | `setupKinematicsTorch` (~410 lines of transcribed BMJ12 kinematics), the non-batch tensor CFF layer (`cffTensor`, `C_VCS0`×3, `C_I0`, `C_I0n`, `S_I0n`), the non-batch cross-section methods, the `zeroC()` helper, and all non-batch cached `double` members. **`m_M` kept** — proton mass, a constant the *batched* setup and CFF layer also read (the one unsuffixed member shared by both). 1350 → 718 lines. |
| `DVCSAluMinusTorch.{h,cpp}` | `aLUTensor`, and the now-unused `kF64` constant |

**This cut a third full copy of the BMJ12 transcription.** Before: base scalar (PARTONS, doubles, full coverage) + non-batch tensor + batched tensor — the two tensor copies being line-for-line parallel, differing only in shape. After: two. That reaches the 2026-06-22 note's stated goal (*"collapsing the single-point-tensor and batched-tensor copies into one"*) by removal rather than by the shape-polymorphic unification originally planned.

### Behavioral change: bare `DVCSAluMinusTorch` now throws

`computeTensorImpl` previously worked via `aLUTensor`; it is now a thin N=1 wrapper over `computeTensorImplBatch`, which is still the throwing placeholder. So `computeTensor` **and** the inherited scalar `computeObservable` both throw for a bare `DVCSAluMinusTorch`. Acceptable because nothing instantiates it, and the planned raw-per-φ leaf (2026-09-15 open task) is specified to build on the `…Batch` chain anyway — looping the single-point `aLUTensor` N times would have reinstated exactly the per-point loop the batching work existed to remove. The throw disappears when that leaf gets its own-φ implementation; at that point `DVCSAluMinusTorch` becomes a valid scalar drop-in for A_LU again (N=1 wrapper reproduces pointwise A_LU at the kinematic's stored φ — *provided* the batch hook uses own-φ semantics, not the shared-φ `[N,M]` convention).

### Verification

`make Run_CFF_NN_Fit` clean. Full `Run_CFF_NN_Fit` run (central fit + `train_replicas(10)` + exports) completed; the three cross-path values agree to every printed digit:

| Path | A_LU^{sin1φ} |
|---|---|
| `observ_calc()` (base PARTONS, scalar) | **0.13186** |
| `observ_calc_torch()` (tensor path) | **0.13186**, `requires_grad = true` |
| `observ_calc_torch_scalar()` (torch scalar virtuals) | **0.13186** |

Note what this check actually is: `observ_calc_torch_scalar()` still routes *through torch* (the leaf's `computeObservable` wraps `computeTensor().item()`), so it is a genuine native-vs-torch differential test between two independent implementations of BMJ12 — not a tautology. That property is what protects against the surviving duplication: nothing links the base scalar and tensor transcriptions, so a PARTONS upgrade or coefficient fix would silently desync them, and this comparison is the only thing that would notice. Its limit is that it is **one** kinematic point in the unpolarized sector; spot-checking a few points spanning the dataset's range would be cheap insurance.

### Reference notes established this session (no code change)

- **How `DVCSProcessBMJ12Torch` stays scalar-compatible**: not by wrapping — by *not overriding*. It never declares `CrossSectionBH/VCS/Interf`, so the vtable still points at PARTONS' implementations. Scalar entry is `compute(...)` → `setKinematics` + `initModule` → inherited virtuals reading the base's **private** doubles; tensor entry is `prepareTensorBatch(...)` → `setupKinematicsTorchBatch` → `*TensorBatch` reading our `m_*Batch`. The scalar path never reaches `setupKinematicsTorchBatch` and never constructs a tensor. Contrast the *leaves* (`DVCSCFFNNTorch::computeCFF`, `DVCSAluMinusTorch::computeObservable`), which **do** use the thin-wrapper pattern and therefore reroute the scalar path into torch physics.
- **Why the `dynamic_cast`s are `dynamic_cast`**: `DVCSObservableTorch` (= `ObservableTorch<K>`) and `DVCSProcessModuleTorch` inherit from *nothing* in PARTONS, so going from a `DVCSObservable*`/`DVCSProcessModule*` to the torch interface is a **cross-cast** between unrelated base subobjects of one complete object — `static_cast` cannot express it. (The `dynamic_cast<DVCSCFFNNTorch*>` is different: a plain downcast, since `DVCSCFFNNTorch` is a single-inheritance PARTONS subclass.) The factory returns `DVCSObservable*` regardless of `classId`, since PARTONS knows nothing about the out-of-tree mixin — hence one base-typed pointer per hierarchy, kept as `CustomLossImpl` members so the chain is wired once and reused every epoch.
- **Tensor-vs-scalar rule for the batch port**: something must be a `torch::Tensor` if it varies along a vectorized axis (N data points **or** M φ nodes) **or** must carry a gradient; otherwise keep it a plain `double` so it broadcasts for free and folds at compile time (`std::sqrt(2.)`, `PI`, `m_M`, and `beamHelicity`/`beamCharge` — which differ *between* calls but are constant *across* the batch). Kinematics tensors are deliberately **no-grad**: they are fixed measured data, not optimization targets, so they enter as coefficients multiplying the grad-carrying CFF tensors. Only the NN parameters are grad leaves (automatic via `torch::nn::Module`); propagation is contagious from there, and the `requires_grad = true` print is the check that nothing (`.item()`, `.detach()`, `NoGradGuard`) severed it.
- **Porting scalar→batch formulae**: the algebra is unchanged; what needs work is (1) **broadcasting** — per-point `[N]` quantities need `unsqueeze(1)` → `[N,1]` before meeting the `[M]` φ axis (the `bc` lambda), and a missed one yields a silent wrong outer product rather than a compile error; (2) **zero-init** — a default-constructed `torch::Tensor` is *undefined*, not zero, so the 72 angular-coefficient slots must be filled explicitly at runtime (a `double[3][3][4] = {}` cannot carry over); (3) `std::`→`torch::` math *selectively* (constants stay `std::`); (4) raw `[N]` tensors replace the bean + `setKinematics`; (5) the batched CFF call passes **xB directly**, dropping the non-batch path's xB→ξ→xB round-trip through `setupKinematicsTorch`.

### Reviewed and deliberately not changed

- **`m_preparedBatch` is a coarse guard, and that is fine.** It records "has `prepareTensorBatch` ever been called," never resets, and so cannot detect an assemble call that doesn't match the cached batch. This was examined at length and judged **not** a hazard worth pre-empting: the only way to trigger it is code that unpacks its kinematics and then never passes them to the process module — visibly broken data flow, an unused-variable warning, and instantly wrong χ² on the first run. A token/generation-counter fix was drafted and rejected as ceremony around a two-line prepare→assemble sequence. If threading is ever added over a shared process-module instance, the flag is the least of it — every `m_*Batch` member and the cached CFF tensors are unsynchronized, and the 2026-06-22 conclusion (clone-per-thread) is the answer.
- **`predict()` still loops per point** with `computeSingleKinematicTorch` while `CustomLoss` uses the batched `computeManyKinematicTorch`. Harmless (runs once per fit, not per epoch); worth switching only if touching that function anyway.

### `train_replicas()` now writes the last replica's learning curve

`CFF_plots_ALU_2007_xpow_replica.ipynb` had gained a cell reading `cff_learning_curve_last_replica.csv`, but `train_replicas()` passed `""` for every replica, so nothing produced it — the copy on disk was a stale Sep-10 artifact, and `*.csv` is gitignored, so the cell could not reproduce from a clean run. Fixed in `train_replicas()`: the last replica (`r == n_replicas-1`) now passes a real path; the rest still pass `""`. Retries need no special handling — `fit_once()` opens with `std::ios::trunc` and the kept model is always the last attempt run, so the surviving file matches the kept replica.

Note this adds a **third** copy of the hardcoded `out_dir` absolute path (alongside `train_nn()` and `predict()`), following the file's existing convention. Worth hoisting to one constant if these paths are ever touched again — and they must be, on any environment move.

Verified on a 10-replica run: the file is rewritten (812 rows, every 2 epochs through 1622), and replica 9 early-stopped at epoch 1623 with best val χ²/n **5.12**. Worth noting what that curve shows — train falls to 0.62 while val climbs to 8.28, i.e. **pronounced overfitting well before early stopping fires**. Best val was reached around epoch ~620 and `patience = 1000` then ran another ~1000 epochs uphill. On a 16-point dataset split into train/val that is not surprising, but it means (a) replica χ²/n values sit far above the smearing noise floor, so the replica band is likely wider than the data alone justifies, and (b) `patience = 1000` (raised from 200 on 2026-09-01) may now be overshooting. Untested hypotheses — flagged for whoever tunes the replica hyperparameters next.

### Open tasks (carried forward)

All 2026-09-01 items stand, plus the 2026-09-15 raw-per-φ dataset task — now the *only* thing blocking `DVCSAluMinusTorch` from being usable at all, since its placeholder is reached by both its batch and (via the N=1 wrapper) single-kinematic hooks.

---

## Session notes (2026-09-17)

### Dead single-point CFF methods removed from `DVCSCFFNNTorch`

Follow-up cleanup to `82edefd` (2026-09-16). That commit deleted the single-point tensor path in the *process* module, which left three methods in the CFF module with no callers anywhere in the repo:

| Removed | Was |
|---|---|
| `computeAllCFFsTensor()` + the `AllCFFsTensor` struct | public; all four CFFs as 0-d tensors, an N=1 wrapper over `computeAllCFFsTensorBatch`. Its last caller was `DVCSProcessBMJ12Torch::setupKinematicsTorch`, deleted in `82edefd` |
| `forwardNN()` | private; the `[1,3]` NN forward (scaling + `xB^m_xPow`), N=1 twin of `forwardNNBatch` |
| `cffComponentTensor()` | private; 0-d Re/Im → complex assembly, N=1 twin of `cffComponentTensorBatch` |

**What survives and why:** `computeCFFTensor(type)` — also an N=1 wrapper, but genuinely reachable: the scalar `computeCFF()` calls it one GPD type at a time, and PARTONS' base-scalar pipeline calls *that* (this is the `observ_calc()` path). It goes straight to `computeCFFTensorBatch`, so `forwardNNBatch`/`cffComponentTensorBatch` are now the only forward and the only Re/Im assembly in the class — i.e. `CFF = xB^m_xPow * NNet_output` is applied in exactly one place (`forwardNNBatch`), where the 2026-08-05 notes say `forwardNN()`.

**Stale comments fixed.** The section header above `computeAllCFFsTensor` claimed it was "still independently called ... by the process module's own single-point setup" — untrue since `82edefd`. Several doc comments described themselves as the "batched sibling of" a method being removed in the same edit; they were reworded to stand alone.

`make Run_CFF_NN_Fit` builds clean. Nothing reachable changed, so the three `observ_calc*` values cannot move and the run was not repeated — a full run retrains and would overwrite the current `cff_model.json` / `obs_*.csv` / replica exports.

### README brought up to date (same session)

`README.md` had drifted ~3 months behind. Corrected: the section claiming batching was "PLANNED, NOT YET IMPLEMENTED" (it landed 2026-09-15) and the run instruction `cd bin && ./Run_CFF_NN_Fit` (wrong — the paths *inside* `bin/partons.properties` resolve against the CWD, so it must be `./bin/Run_CFF_NN_Fit` from the project root; the **Build** section at the top of this file carried the same error and was fixed too). Added the missing 2026-08-05 / 09-01 / 09-15 / 09-16 work, refreshed the file tree, output-file table and data-format section, and added a "Current status / open items" list.

---

## Session notes (2026-09-18)

### Train/eval mode is the caller's, not the forward's

`DVCSCFFNNTorch::forwardNNBatch` called `m_net->eval()` unconditionally — a leftover from when the class was inference-only. Every path through the NN forward therefore ran in eval mode, including the training step, so `fit_once`'s `net->train()` was undone before it could take effect. No effect today (`Linear`/`tanh` ignore the flag), but adding a `Dropout` or `BatchNorm` layer would have made training silently run in inference mode.

Removed that `eval()`; each caller now declares its own mode via **`EvalModeGuard`** (new, in `CFF_NN_Fit.h`): RAII, sets eval on construction and restores the previous mode on scope exit. The restore matters because the `CFFNNModel` is shared **by handle** between the fitter, `CustomLoss` and `DVCSCFFNNTorch` — a bare `eval()` in an inference call would otherwise leak into training that runs afterwards.

Eval mode and gradient tracking are **independent**, and all four combinations occur here:

| Entry point | eval mode | NoGradGuard |
|---|---|---|
| training step | no (`train()`) | no |
| validation step | yes | yes |
| `predict()` | yes | yes |
| `observ_calc_torch()` | yes | **no** — the point of this path is `requires_grad = true` |
| `observ_calc_torch_scalar()` | yes | yes (inside the leaf's scalar virtual) |
| `observ_calc()` | both, set inside `computeCFF()` |

Verified by a full run: the three cross-path values agree (`observ_calc` = 0.145646, `observ_calc_torch` = `observ_calc_torch_scalar` = 0.145647, tensor path keeps `requires_grad = true`), central fit R² = 0.65, χ²/n = 0.47, all 10 replicas trained. Commit `4253568`.

### Replica retries: 30 tries, then fail loudly

`max_retries_per_replica = 5` → **`max_tries_per_replica = 30`**. The rename fixes a real ambiguity: the loop always counted *total tries*, so the old name promised one more attempt than the code gave.

The bigger change is the exhaustion policy. Before: keep the last hopeless attempt with a warning, so a run always produced exactly `n_replicas` models — one of which could be junk, silently widening the band. Now: **export the replicas accepted so far, then throw** `std::runtime_error`. Rationale: a partial ensemble is not a result, but the compute already spent is worth keeping.

For comparison, Gepard (`fitter_vectloss.py`) has two policies and neither matches: `fit()` retries with **no cap** (`while test_err < 0`) so exhaustion cannot happen; `fitgood()` caps tries **globally** across the ensemble and on exhaustion just `break`s, returning fewer nets than requested with no error. Both discard the failed net and keep the successes — the same as here; only the ending differs.

**`export_replicas` now deletes the previous `<prefix>*.json` before writing** (unless there is nothing to write). Without it, a 10-replica run followed by a 5-replica run left `_05`…`_09` on disk, and a `glob` in the plotting notebook would read 10 replicas of which 5 were from a different fit. This was the concrete hazard that motivated exporting-then-throwing rather than writing to a distinct prefix.

**Verified on three real runs** (temporary `train_replicas(...)` args in `Run_CFF_NN_Fit.cpp`, reverted after):

| Scenario | Result |
|---|---|
| all tries fail at replica 0 (`10, 2, 0.5f, 2`) | `No replicas to export; leaving … untouched` → throw; the previous 10 JSONs survived |
| shorter ensemble succeeds (`4, 2, 6.0f, 200`) | `Removed 10 …` → `Exported 4`; directory holds 4, not 10 |
| mid-ensemble failure (`6, 2, 4.5f, 200`) | replicas 0–2 accepted, replica 3 exhausted → `Removed 4` → `Exported 3` → throw naming replica 3 |

### Open task: the process still exits 0 on failure

`main()` catches the exception, logs it through PARTONS' logger and falls through to `return 0`, so a failed run reports success to the shell — SWIF/Slurm marks the job succeeded, and `./bin/Run_CFF_NN_Fit && …` continues onto an incomplete ensemble. The farm `.out` file does end with the `[ERROR] (main::main) Replica N still hopeless …` line (two lines above `Total run time`), so a human reading the log sees it; automation does not.

Fix when convenient: an `int exit_code` set to 1 in both catch blocks and returned at the end — keeping the timing print, and incidentally removing the double `pPartons->close()` that the error path currently performs (once in the catch, once after the try/catch).
