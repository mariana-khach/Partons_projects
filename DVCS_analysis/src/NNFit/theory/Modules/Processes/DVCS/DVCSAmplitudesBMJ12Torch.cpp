/**
 * @file DVCSAmplitudesBMJ12Torch.cpp
 *
 * Implements the torch-tensor VCS and interference amplitudes following
 * BMJ12 (arXiv:1212.6674), restricted to:
 *   • unpolarised target (Λ = 0)
 *   • electron beam (charge = –1)
 *   • leading-twist CFFs only (CFF_FT = CFF_FLT = 0)
 *
 * Physics summary:
 *   σ_VCS   bilinear in CFFs through C_VCS(S=0) → torch tensor, φ-independent
 *   σ_Interf linear in CFFs through C_I(S=0)   → torch tensor, but prefactor
 *            has φ-dependence via lepton propagators P1(φ)·P2(φ)
 *   σ_BH    no CFF dependence                  → plain double
 *
 * For the unpolarised, CFF_FT=0 case, Im(C_VCS^{S=0}) = 0.
 * The only λ-odd contribution therefore comes from the interference term
 * through sI[n] ∝ Im(CFF combinations).
 */

#include "NNFit/theory/Modules/Processes/DVCS/DVCSAmplitudesBMJ12Torch.h"
#include <cmath>

namespace Theory {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: zero 0-d torch tensor with grad enabled
// ─────────────────────────────────────────────────────────────────────────────
static torch::Tensor zeros_like_scalar(const torch::Tensor& t)
{
    return torch::zeros({}, t.options());
}

// ─────────────────────────────────────────────────────────────────────────────
// computeDressedCFFs
// ─────────────────────────────────────────────────────────────────────────────

DressedCFFs computeDressedCFFs(
    const DVCSKin&       kin,
    const torch::Tensor& H_re,  const torch::Tensor& H_im,
    const torch::Tensor& E_re,  const torch::Tensor& E_im,
    const torch::Tensor& Ht_re, const torch::Tensor& Ht_im,
    const torch::Tensor& Et_re, const torch::Tensor& Et_im)
{
    DressedCFFs dc;
    dc.H_re  = H_re;   dc.H_im  = H_im;
    dc.E_re  = E_re;   dc.E_im  = E_im;
    dc.Ht_re = Ht_re;  dc.Ht_im = Ht_im;
    dc.Et_re = Et_re;  dc.Et_im = Et_im;

    for (int j = 0; j < 3; ++j) {
        const double c = kin.cF[j];
        dc.FH_re[j]  = c * H_re;   dc.FH_im[j]  = c * H_im;
        dc.FE_re[j]  = c * E_re;   dc.FE_im[j]  = c * E_im;
        dc.FHt_re[j] = c * Ht_re;  dc.FHt_im[j] = c * Ht_im;
        dc.FEt_re[j] = c * Et_re;  dc.FEt_im[j] = c * Et_im;
    }
    return dc;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeVCSCoeffs
// ─────────────────────────────────────────────────────────────────────────────
// For an unpolarised target with CFF_FT=CFF_FLT=0, every C_VCS(S=0) call
// is real, so Im(C_VCS(S=0)) = 0 and sVCS[0][1] = 0.
// The three non-zero coefficients c0, c1, c2 are real, bilinear in the raw
// CFF absolute values / cross products.
//
// We factor out the common body (C_VCS_raw) and multiply by the appropriate
// product of cF coefficients assembled from the Fourier formulae:
//
//   c0 = B · [2·B2·(c0²+c1²) + 8·C2·c2²] / er1 · CVCSraw
//   c1 = B · 4√2·C1·(2–y)·c2·(c0+c1) / er1 · CVCSraw
//   c2 = B · 8·C2·c0·c1 / er1 · CVCSraw
//
// where CVCSraw = 4(1–xB)(1+xBtQ2)/xBtQ2_1 · (|H|²+|Ht|²)
//              + (2+tQ2)ε²/xBtQ2_1 · |Ht|²
//              – (t/4M²) · |E|²
//              – xB²/xBtQ2_1 · [(1+tQ2)² · 2Re(H·E*) + (1+tQ2)²·|E|²
//                                + 2Re(Ht·Et*) + (t/4M²)|Et|²]
// ─────────────────────────────────────────────────────────────────────────────

VCSCoeffs computeVCSCoeffs(const DVCSKin& kin, const DressedCFFs& dc)
{
    VCSCoeffs r;
    r.prefVCS = E6 / (kin.y2 * kin.Q2);

    const double xB    = kin.xB;
    const double tQ2   = kin.tQ2;
    const double dM    = kin.Delta2M2;
    const double eps2  = kin.eps2;
    const double er1   = kin.er[1];
    const double x1    = kin.xBtQ2_1;

    // ── CFF absolute-value squares ────────────────────────────────────────
    const torch::Tensor H2  = dc.H_re * dc.H_re  + dc.H_im  * dc.H_im;   // |H|²
    const torch::Tensor E2  = dc.E_re * dc.E_re  + dc.E_im  * dc.E_im;   // |E|²
    const torch::Tensor Ht2 = dc.Ht_re* dc.Ht_re + dc.Ht_im * dc.Ht_im; // |Ht|²
    const torch::Tensor Et2 = dc.Et_re* dc.Et_re + dc.Et_im * dc.Et_im;  // |Et|²

    // ── Re(H·E*)  and  Re(Ht·Et*) ────────────────────────────────────────
    const torch::Tensor ReHE  = dc.H_re * dc.E_re  + dc.H_im * dc.E_im;
    const torch::Tensor ReHtEt= dc.Ht_re* dc.Et_re + dc.Ht_im* dc.Et_im;

    // ── C_VCS raw body (S=0, unscaled) ───────────────────────────────────
    const double a1 = 4.*(1.-xB)*(1. + xB*tQ2) / x1;
    const double a2 = (2.+tQ2)*eps2 / x1;
    const double a3 = -tQ2 * kin.Q2 / (4.*kin.M2);  // = –t/(4M²)  with t=tQ2*Q2
    const double a4 = -kin.xB2 / x1 * std::pow(1.+tQ2, 2);
    const double a5 = -kin.xB2 / x1;
    const double a6 = -kin.xB2 / x1 * dM;

    const torch::Tensor CVCSraw =
          a1*(H2 + Ht2)
        + a2*Ht2
        + a3*E2
        + a4*(2.*ReHE + E2)
        + a5*2.*ReHtEt
        + a6*Et2;

    // ── Kinematic scale factors from VCS Fourier formulae ─────────────────
    const double c0 = kin.cF[0], c1_f = kin.cF[1], c2_f = kin.cF[2];
    const double y    = kin.y;
    const double C1_k = kin.C1, C2_k = kin.C2;
    const double B2   = kin.B2;

    const double k0 = (2.*B2*(c0*c0 + c1_f*c1_f) + 8.*C2_k*c2_f*c2_f) / er1;
    const double k1 = 4.*std::sqrt(2.)*C1_k*(2.-y)*c2_f*(c0+c1_f) / er1;
    const double k2 = 8.*C2_k*c0*c1_f / er1;

    r.c0 = k0 * CVCSraw;
    r.c1 = k1 * CVCSraw;
    r.c2 = k2 * CVCSraw;

    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeInterfCoeffs
// ─────────────────────────────────────────────────────────────────────────────
// The interference amplitude is linear in the dressed CFFs:
//
//   C_I(S=0, a,b, "")  = cF[j] · (F1·H – (t/4M²)F2·E + xBF1pF2·Ht)
//   C_I(S=0, a,b, "V") = cF[j] · xBF1pF2 · (H + E)
//   C_I(S=0, a,b, "A") = cF[j] · xBF1pF2 · Ht
//
// where xBF1pF2 = xB·(F1+F2)/xBtQ2_0.
//
// The cosine (λ-even) and sine (λ-odd) Fourier coefficients are then:
//
//   cI[n] = Re { Σ_j cF[j] · [C[j][0][n]·CI_raw + C[j][1][n]·CIV_raw + C[j][2][n]·CIA_raw] }
//   sI[n] = Σ_j cF[j] · [S[j][0][n]·Im(CI_raw) + S[j][1][n]·Im(CIV_raw) + S[j][2][n]·Im(CIA_raw)]
//            (sI[n] has a global λ factor added in crossSectionAtPhi)
// ─────────────────────────────────────────────────────────────────────────────

InterfCoeffs computeInterfCoeffs(const DVCSKin& kin, const DressedCFFs& dc)
{
    InterfCoeffs r;
    // Initialise sI[0] and sI[3] to zero (not accumulated below)
    auto zero = zeros_like_scalar(dc.H_re);
    r.sI[0] = zero;
    r.sI[3] = zero;
    r.cI[0] = r.cI[1] = r.cI[2] = r.cI[3] = zero;
    r.sI[1] = r.sI[2] = zero;

    const double xBF1pF2 = kin.xB * kin.F1PlusF2 / kin.xBtQ2_0;

    // ── Raw C_I(S=0) bodies  (Re and Im parts) ────────────────────────────
    //   CI_raw  = F1·H  – (t/4M²)·F2·E + xBF1pF2·Ht
    const torch::Tensor CI_re  = kin.F1*dc.H_re
                                - kin.Delta2M2*kin.F2*dc.E_re
                                + xBF1pF2*dc.Ht_re;
    const torch::Tensor CI_im  = kin.F1*dc.H_im
                                - kin.Delta2M2*kin.F2*dc.E_im
                                + xBF1pF2*dc.Ht_im;

    //   CIV_raw = xBF1pF2·(H + E)
    const torch::Tensor CIV_re = xBF1pF2*(dc.H_re + dc.E_re);
    const torch::Tensor CIV_im = xBF1pF2*(dc.H_im + dc.E_im);

    //   CIA_raw = xBF1pF2·Ht
    const torch::Tensor CIA_re = xBF1pF2*dc.Ht_re;
    const torch::Tensor CIA_im = xBF1pF2*dc.Ht_im;

    // ── Accumulate over the three helicity combinations j=0,1,2 ──────────
    for (int j = 0; j < 3; ++j) {
        const double cf = kin.cF[j];
        for (int n = 0; n < 4; ++n) {
            const double w0 = kin.C_ang[j][0][n];
            const double w1 = kin.C_ang[j][1][n];
            const double w2 = kin.C_ang[j][2][n];
            // cI[n] uses real parts only (cosine Fourier series = λ-even part)
            r.cI[n] = r.cI[n]
                    + cf*(w0*CI_re + w1*CIV_re + w2*CIA_re);
        }
        for (int n = 1; n <= 2; ++n) {
            const double w0 = kin.S_ang[j][0][n];
            const double w1 = kin.S_ang[j][1][n];
            const double w2 = kin.S_ang[j][2][n];
            // sI[n] uses imaginary parts (sine series = λ-odd, λ factor added later)
            r.sI[n] = r.sI[n]
                    + cf*(w0*CI_im + w1*CIV_im + w2*CIA_im);
        }
    }
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// crossSectionAtPhi
// ─────────────────────────────────────────────────────────────────────────────
// σ(λ, φ) = phaseSpace · [BH(φ) + VCS(φ) + λ·VCS_odd(φ)  [= 0 here]
//                        + A_I(φ)·Σ_n cI[n]·cos(n φ₁)
//                        + λ·A_I(φ)·Σ_n sI[n]·sin(n φ₁) ]
//
// where φ₁ = π – φ  (Trento convention)
//   BH(φ)   = e6/(xB² y² er3 t P1 P2) · Σ_n cBH[n]·cos(n φ₁)
//   VCS(φ)  = prefVCS · Σ_n cVCS[n]·cos(n φ₁)      [φ-independent prefactor]
//   A_I(φ) = e6/(xB y³ t P1(φ) P2(φ))              [φ-dependent via P1,P2]
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor crossSectionAtPhi(
    const DVCSKin&      kin,
    const VCSCoeffs&    vcs,
    const InterfCoeffs& interf,
    double              phi,
    double              beamHelicity)
{
    // ── Angles ──────────────────────────────────────────────────────────────
    // phi1 = π – φ  (BMK small-angle between electron plane and proton plane)
    const double phi1    = PI - phi;
    const double cos1    = std::cos(phi1);
    const double cos2    = std::cos(2.*phi1);
    const double sin1    = std::sin(phi1);
    const double sin2    = std::sin(2.*phi1);
    const double cos3    = std::cos(3.*phi1);
    // note: sin(n*phi1) where phi1=π-φ  so sin(phi1)=sin(φ), sin(2phi1)=-sin(2φ)…
    // but we use the Trento φ for the GL integration sin(φ), so we keep
    // everything in phi1 = π-φ coordinates consistently.
    //
    // Also:  cos(phi1) = cos(π-φ) = –cos(φ),  sin(phi1) = sin(π-φ) = sin(φ)
    // for reference – but the Fourier series is written in phi1, so no
    // transformation is needed here.

    // ── Lepton propagators: P1(φ) = P1_c – P1_K·cos(φ) ──────────────────
    // cos(phi1) = –cos(φ)  →  P1 = –J/yeps – 2K·cos(phi1)/yeps
    //                            = P1_c – P1_K·(–cos(φ)) ... but:
    // From DVCSProcessBMJ12: P1 = –(J + 2K·cos(phi1))/yeps
    // P1_c = –J/yeps,  P1_K = 2K/yeps (coefficient of cos(phi1) with extra minus sign absorbed)
    // → P1 = P1_c – P1_K·cos(phi1)   ... but we stored P1_K positively
    //   P1 = –J/yeps – P1_K·cos(phi1)
    // Actually let's re-derive:  P1 = –(J + 2K·cos(phi1))/yeps = –J/yeps – 2K·cos(phi1)/yeps
    // We stored: P1_c = –J/yeps,  P1_K = 2K/yeps
    // So:        P1 = P1_c – P1_K·cos(phi1)
    const double P1 = kin.P1_c - kin.P1_K * cos1;
    const double P2 = kin.P12_sum - P1;          // P1 + P2 = 1 + t/Q2

    // ── BH amplitude (no torch – pure double) ─────────────────────────────
    // m_epsroot[3] in BMJ12 = (1+ε²)²  = er[3]  (not er[2] = (1+ε²)^{3/2})
    const double prefBH = E6 / (kin.xB2*kin.y2*kin.er[3]*kin.t * P1*P2);
    const double bh_amp = prefBH
                        * (kin.cBH[0]
                         + kin.cBH[1]*cos1
                         + kin.cBH[2]*cos2);

    // ── VCS amplitude (torch, φ-independent prefactor) ────────────────────
    // sVCS[0][1] = 0 for unp target + CFF_FT=0, so no λ-odd VCS term
    const torch::Tensor vcs_amp = vcs.prefVCS
                                * (vcs.c0
                                 + vcs.c1*cos1
                                 + vcs.c2*cos2);

    // ── Interference amplitude (torch, φ-dependent prefactor) ─────────────
    // Beam charge = –1 →  –beamCharge = +1  (factor already +1 in prefI)
    const double prefI = E6 / (kin.xB * kin.y3 * kin.t * P1*P2);

    const torch::Tensor interf_cos =
          interf.cI[0]
        + interf.cI[1]*cos1
        + interf.cI[2]*cos2
        + interf.cI[3]*cos3;

    const torch::Tensor interf_sin =
          interf.sI[1]*sin1
        + interf.sI[2]*sin2;

    const torch::Tensor interf_amp = prefI
                                   * (interf_cos + beamHelicity*interf_sin);

    // ── Total cross-section ───────────────────────────────────────────────
    // phaseSpace factors appear in both numerator and denominator of A_LU
    // and cancel; we keep them for correctness but could omit.
    return kin.phaseSpace * (bh_amp + vcs_amp + interf_amp);
}

} // namespace Theory
