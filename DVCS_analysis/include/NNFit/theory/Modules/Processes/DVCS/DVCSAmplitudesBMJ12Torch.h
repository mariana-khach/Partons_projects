#pragma once

/**
 * @file DVCSAmplitudesBMJ12Torch.h
 *
 * Torch-tensor implementation of the BMJ12 DVCS cross-section for an
 * unpolarised target and an electron beam.
 *
 * The key design goal is differentiability: CFF values arrive as
 * torch::Tensor scalars (0-d) carrying grad information from the NN,
 * and all subsequent arithmetic stays inside the autograd graph.
 *
 * Twist-3 / helicity-flip CFFs (CFF_FT, CFF_FLT) are set to zero;
 * only the four leading-twist CFFs H, E, H̃, Ẽ are used.
 *
 * Public API:
 *   struct DressedCFFs    – result of dressing raw NN outputs
 *   struct VCSCoeffs      – Fourier coefficients for the VCS² amplitude
 *   struct InterfCoeffs   – Fourier coefficients for the BH-DVCS interference
 *
 *   computeDressedCFFs(kin, H_re, H_im, E_re, E_im, Ht_re, Ht_im, Et_re, Et_im)
 *   computeVCSCoeffs(kin, dc)
 *   computeInterfCoeffs(kin, dc)
 *   crossSectionAtPhi(kin, bh_n, vcs, interf, phi, beamHelicity)
 */

#include "NNFit/theory/Beans/Obs/DVCS/DVCSKinematicsTorch.h"
#include <torch/torch.h>

namespace Theory {

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Dressed CFFs  F̂_X(j) = cF[j] · X_raw  (j = 0,1,2).
 * Stored as (real, imaginary) pairs; all are 0-d torch Tensors
 * connected to the autograd graph.
 *
 * For the computation, only the bilinear products that appear in the
 * VCS amplitude and the linear combinations that appear in the
 * interference amplitude need to be kept in the graph.
 */
struct DressedCFFs {
    // Raw NN outputs (0-d tensors)
    torch::Tensor H_re,  H_im;
    torch::Tensor E_re,  E_im;
    torch::Tensor Ht_re, Ht_im;
    torch::Tensor Et_re, Et_im;

    // Scaled copies  F_X[j] = cF[j] * X_raw  (j=0,1,2)
    // Re and Im parts separately
    torch::Tensor FH_re[3],  FH_im[3];
    torch::Tensor FE_re[3],  FE_im[3];
    torch::Tensor FHt_re[3], FHt_im[3];
    torch::Tensor FEt_re[3], FEt_im[3];
};

/**
 * Fourier coefficients of the VCS² amplitude (φ₁ = π–φ).
 *
 *  σ_VCS(φ) = prefVCS * [c0·cos(0) + c1·cos(φ₁) + s1·sin(φ₁) + c2·cos(2φ₁)]
 *
 * prefVCS = e⁶ / (y² Q²)  (stored separately, φ-independent)
 *
 * For an unpolarised target with CFF_FT=CFF_FLT=0:
 *   s1 is always zero (Im(C_VCS^{S=0}) vanishes).
 *   c0, c1, c2 are real torch tensors (bilinear in |CFF|² and Re(H E*) etc.)
 */
struct VCSCoeffs {
    torch::Tensor c0, c1, c2;   // cosine coefficients (torch scalars)
    double prefVCS;              // e6 / (y² Q²)
};

/**
 * Fourier coefficients of the BH–DVCS interference amplitude.
 *
 *  σ_I(λ,φ) = A_I(φ) * [ Σ_{n=0}^{3} cI[n]·cos(n φ₁)
 *                        + λ · Σ_{n=1}^{2} sI[n]·sin(n φ₁) ]
 *
 * A_I(φ) = e⁶ / (x_B y³ t P1(φ) P2(φ))   (φ-dependent; computed inline)
 *
 * cI[n] ∝ Re(CFF combinations) – stays in grad graph via H_re, E_re etc.
 * sI[n] ∝ Im(CFF combinations) – stays in grad graph via H_im, E_im etc.
 * sI[0] = sI[3] = 0 by convention (not computed).
 */
struct InterfCoeffs {
    torch::Tensor cI[4];   // cosine coefficients (n=0..3)
    torch::Tensor sI[4];   // sine   coefficients (n=0..3); sI[0]=sI[3]=0
};

// ─────────────────────────────────────────────────────────────────────────────
// Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Dress the eight raw NN CFF outputs with kinematic helicity-combination
 * coefficients cF[j] from the precomputed DVCSKin struct.
 */
DressedCFFs computeDressedCFFs(
    const DVCSKin&     kin,
    const torch::Tensor& H_re,  const torch::Tensor& H_im,
    const torch::Tensor& E_re,  const torch::Tensor& E_im,
    const torch::Tensor& Ht_re, const torch::Tensor& Ht_im,
    const torch::Tensor& Et_re, const torch::Tensor& Et_im);

/**
 * Compute the VCS² Fourier coefficients as torch tensors.
 * Bilinear in the dressed CFFs, hence bilinear in the raw NN outputs.
 */
VCSCoeffs computeVCSCoeffs(const DVCSKin& kin, const DressedCFFs& dc);

/**
 * Compute the BH–DVCS interference Fourier coefficients as torch tensors.
 * Linear in the dressed CFFs, hence linear in the raw NN outputs.
 */
InterfCoeffs computeInterfCoeffs(const DVCSKin& kin, const DressedCFFs& dc);

/**
 * Evaluate the total cross-section  σ(λ, φ)  at a single azimuthal angle.
 *
 * @param kin          pre-computed kinematics
 * @param vcs          VCS Fourier coefficients (torch)
 * @param interf       Interference Fourier coefficients (torch)
 * @param phi          azimuthal angle in Trento convention [rad]
 * @param beamHelicity ±1
 * @return             σ as a torch scalar (with grad)
 */
torch::Tensor crossSectionAtPhi(
    const DVCSKin&       kin,
    const VCSCoeffs&     vcs,
    const InterfCoeffs&  interf,
    double               phi,
    double               beamHelicity);

} // namespace Theory
