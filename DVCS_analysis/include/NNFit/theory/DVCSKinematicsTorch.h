#pragma once

/**
 * @file DVCSKinematicsTorch.h
 *
 * Pre-computes all φ-independent kinematic quantities needed by the
 * BMJ12 cross-section formulae for an unpolarised proton target and
 * an electron beam.  These are plain C++ doubles – no torch dependency.
 *
 * Notation follows Belitsky–Mueller–Ji (2012), arXiv:1212.6674.
 *
 * Physical constants match those used in PARTONS (FundamentalPhysicalConstants.h):
 *   M_p   = 0.938272013 GeV
 *   e     = 0.30282211985966434  (e^2 = 4π α)
 */

#include <cmath>

namespace Theory {

// ── Physical constants ────────────────────────────────────────────────────────
constexpr double M_PROTON        = 0.938272013;
constexpr double POSITRON_CHARGE = 0.30282211985966434;
constexpr double PI              = 3.14159265358979323846;
/// e^6  (used as overall coupling in all three cross-section terms)
constexpr double E6 = POSITRON_CHARGE * POSITRON_CHARGE * POSITRON_CHARGE
                    * POSITRON_CHARGE * POSITRON_CHARGE * POSITRON_CHARGE;

// ── Kinematic struct ─────────────────────────────────────────────────────────
/**
 * All BMJ12 kinematic quantities derived from a single set
 * (xB, t, Q2, E).  Compute once per kinematic point; reuse for the
 * full φ-integration and for both beam-helicity signs.
 *
 * Arrays match PARTONS index conventions, e.g.
 *   er[0] = sqrt(1+ε²),  er[1] = 1+ε²,  er[2] = (1+ε²)^{3/2}, …
 */
struct DVCSKin {

    // ── Primary inputs ──────────────────────────────────────────────────
    double xB, t, Q2, E;

    // ── Proton mass ─────────────────────────────────────────────────────
    double M, M2;

    // ── Inelasticity and epsilon ────────────────────────────────────────
    double y, y2, y3;
    double eps;          ///< ε = 2 x_B M / Q
    double eps2;         ///< ε²
    double er[6];        ///< er[k] = (1+ε²)^{(k+1)/2} following PARTONS ordering:
                         ///< er[0]=√(1+ε²), er[1]=1+ε², er[2]=(1+ε²)^{3/2},
                         ///< er[3]=(1+ε²)², er[4]=(1+ε²)^{5/2}, er[5]=(1+ε²)^3

    // ── Photon virtuality powers ────────────────────────────────────────
    double Q;            ///< sqrt(Q2)
    double Q3;           ///< Q^3
    double Q4;           ///< Q^4

    // ── Bjorken variable powers ─────────────────────────────────────────
    double xB2;

    // ── Composite kinematic factor (eq. A1 in BMK notation) ────────────
    double xBtQ2_0;      ///< 2 – x_B + x_B t/Q2
    double xBtQ2_1;      ///< xBtQ2_0²
    double xBtQ2_2;      ///< xBtQ2_0³

    // ── Kinematic limits ─────────────────────────────────────────────────
    double tmin;         ///< minimum (most-negative) allowed t
    double tmax;         ///< maximum allowed t  (≈ 0)
    double tp;           ///< t – tmin  (≥ 0)

    // ── t-ratios ─────────────────────────────────────────────────────────
    double tQ2;          ///< t / Q2  (= Δ²/Q² in BMK; also called Delta2Q2)
    double Delta2M2;     ///< t / (4 M²)
    double Kt2Q2;        ///< K̃² / Q2
    double tpQ2;         ///< (t – tmin) / Q2

    // ── Transverse kinematic factors K, K̃ ──────────────────────────────
    double K2, K;
    double Kt2, Kt;

    // ── Lepton propagator quantities ──────────────────────────────────────
    double J_val;        ///< J = (1–y–y ε²/2)(1+t/Q2) – (1–x_B)(2–y) t/Q2
    double yeps;         ///< y (1 + ε²)
    /// P1(φ) = –(J + 2K cos(π–φ)) / yeps = (J – 2K cos(φ)) / yeps + const
    /// Convenient decomposition: P1(φ) = P1_c + P1_K * cos(φ)
    double P1_c;         ///< –J / yeps   (constant part)
    double P1_K;         ///< 2K / yeps   (coefficient of cos(φ))
    double P12_sum;      ///< 1 + t/Q2    (= P1 + P2)

    // ── Dirac / Pauli form factors (dipole parametrisation from BMJ12) ──
    double F1, F2;
    double F1PlusF2;
    double F1PlusF22;    ///< (F1 + F2)²
    double F1MinusDeltaF2; ///< F1² – (t/4M²) F2²
    double F1PlusDeltaF2;  ///< F1 + (t/4M²) F2  (used in BH LP terms)

    // ── Phase-space prefactor ─────────────────────────────────────────────
    double phaseSpace;   ///< x_B y² / (1024 π⁵ Q⁴ √(1+ε²))

    // ── Useful y-ε combinations ──────────────────────────────────────────
    double C1;           ///< sqrt(1 – y – ε² y²/4)
    double C2;           ///< 1 – y – ε² y²/4
    double B2;           ///< 2 – 2y + y² + ε² y²/2

    // ── Dressed-CFF scaling coefficients c_F[j] ─────────────────────────
    /// For twist-2 leading-helicity CFFs (CFF_FT = CFF_FLT = 0):
    ///   F̂_X(a,b) = cF[j(a,b)] · X_raw
    /// where j=0 ↔ (a,b)=(+1,+1), j=1 ↔ (a,b)=(+1,–1), j=2 ↔ (a,b)=(0,±1).
    double cF[3];

    // ── BH Fourier coefficients (unpolarised target) ──────────────────────
    /// Σ_n cBH[n] cos(n φ₁)  where φ₁ = π – φ (Trento convention)
    double cBH[3];       ///< cBH[0..2] = c₀,c₁,c₂ of the unp BH series

    // ── Angular-coefficient arrays for the interference term ─────────────
    /// C_ang[i][k][n] → m_C in BMJ12 notation.
    ///   i ∈ {0=++, 1=–+, 2=0+}  (virtual-photon helicity combination)
    ///   k ∈ {0=bare, 1=V, 2=A}  (CFF flavour combination)
    ///   n ∈ {0..3}               (Fourier mode)
    double C_ang[3][3][4];
    double S_ang[3][3][4]; ///< S_ang[i][k][n] → m_S in BMJ12
};

// ── Factory ──────────────────────────────────────────────────────────────────
/**
 * Compute the full DVCSKin struct from primary kinematics.
 * Beam charge = –1 (electron), unpolarised target (Λ=0).
 */
DVCSKin computeKinematics(double xB, double t, double Q2, double E);

} // namespace Theory
