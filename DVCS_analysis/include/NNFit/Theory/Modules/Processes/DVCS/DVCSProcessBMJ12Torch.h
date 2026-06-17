//
// Created by Mariana Khachatryan on 6/15/26.
//

#ifndef DVCS_PROCESS_BMJ12_TORCH_H
#define DVCS_PROCESS_BMJ12_TORCH_H

#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/process/DVCS/DVCSProcessBMJ12.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

/**
 * @class DVCSProcessBMJ12Torch
 *
 * @brief Differentiable (libtorch) twin of PARTONS::DVCSProcessBMJ12.
 *
 * Subclasses DVCSProcessBMJ12 so it is a drop-in process module: driven through
 * the scalar pipeline it behaves exactly like the base class (the inherited
 * scalar CrossSection* virtuals + the attached DVCSCFFNNTorch scalar wrapper).
 *
 * For the tensor path it adds crossSectionTensor(): the BMJ12 unpolarized-target
 * cross section sigma(lambda, phi) evaluated batched over a [N] tensor of phi
 * values, with the CFFs taken as complex tensors from DVCSCFFNNTorch so that the
 * autograd graph runs from the NN parameters to the cross section.
 *
 * The pure-kinematic BMJ12 machinery (Fourier/angular coefficients, K, epsilon,
 * form factors, phase space, ...) is transcribed verbatim from
 * DVCSProcessBMJ12.cpp as plain doubles (no gradient); only the CFF-bilinear
 * (VCS) and CFF-linear (interference) layers are evaluated in tensors. Because
 * the observable that uses this (A_LU) has an unpolarized target (Lambda = 0),
 * only the unpolarized Fourier-coefficient sector is ported.
 */
class DVCSProcessBMJ12Torch: public PARTONS::DVCSProcessBMJ12,
        public DVCSProcessModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSProcessBMJ12Torch(const std::string& className);
    virtual ~DVCSProcessBMJ12Torch();

    virtual DVCSProcessBMJ12Torch* clone() const override;

    // Sub-process cross sections sigma_X(lambda, phi), batched over phi (tensor
    // siblings of CrossSectionBH / CrossSectionVCS / CrossSectionInterf). Each
    // assumes the phi-independent setup has run; the base crossSectionTensor()
    // template method drives setup + the selected sum.
    torch::Tensor crossSectionBHTensor(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) override;
    torch::Tensor crossSectionVCSTensor(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) override;
    torch::Tensor crossSectionInterfTensor(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) override;

protected:

    DVCSProcessBMJ12Torch(const DVCSProcessBMJ12Torch& other);

private:

    /**
     * Prepare the phi-independent quantities for the tensor path at the given
     * kinematics: reuse the base protected kinematics, transcribe the BMJ12
     * derived quantities and angular coefficients, push (xi, t, Q2) to the
     * attached DVCSCFFNNTorch and run one NN forward to cache the CFF tensors.
     * Called once by the base crossSectionTensor() template method.
     */
    void setupKinematicsTorch(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    // ----- tensor CFF layer -----------------------------------------------

    /** CFF tensor for GPD index F (0=H,1=E,2=Ht,3=Et) and helicity labels a,b. */
    torch::Tensor cffTensor(int F, int a, int b) const;

    /** S=0 VCS contraction C^{VCS}_0 (bilinear in CFFs), 2 / 3 / 4 pair forms. */
    torch::Tensor C_VCS0(int a1, int b1, int a2, int b2) const;
    torch::Tensor C_VCS0(int a1, int b1, int a2, int b2, int a3, int b3) const;
    torch::Tensor C_VCS0(int a1, int b1, int a2, int b2, int a3, int b3, int a4,
            int b4) const;

    /** S=0 interference contraction C^{I}_0 (linear in CFFs), VA in {"","V","A"}. */
    torch::Tensor C_I0(int a, int b, const std::string& VA) const;
    /** Harmonic-n interference combinations (cosine / sine towers). */
    torch::Tensor C_I0n(unsigned int n, int a, int b) const;
    torch::Tensor S_I0n(unsigned int n, int a, int b) const;

    // ----- cached state ----------------------------------------------------

    // CFFs from the NN (0-d complex double, grad-tracked)
    torch::Tensor m_CFFstd[4];   ///< H, E, Ht, Et.
    torch::Tensor m_CFF[4][3];   ///< m_CFF[F][j] = cF[j]*CFFstd[F], j in {0,1,2}.

    // BMJ12 kinematics (doubles, no gradient) — transcribed from DVCSProcessBMJ12
    double m_xB2;
    double m_Qpow[4];        ///< Q, Q^2, Q^3, Q^4.
    double m_Delta2[2];      ///< t, t^2.
    double m_xBtQ2[3];
    double m_M[2];           ///< M, M^2.
    double m_yBMJ[3];
    double m_epsilonBMJ[2];
    double m_epsroot[6];
    double m_K[2];
    double m_Kt[2];
    double m_tminBMJ;
    double m_tmaxBMJ;
    double m_F1, m_F2;
    double m_phaseSpaceBMJ;
    double m_J;              ///< Numerator of lepton propagator P1.
    double m_yeps;           ///< y (1 + eps^2).
    double m_cF[3];          ///< cF[j][0] coefficients (j = 0,1,2).
    double m_cBH0[3];        ///< Unpolarized BH Fourier coeffs c0,c1,c2.
    double m_Cang[3][3][4];  ///< Interference angular coeffs C (i, k, n).
    double m_Sang[3][3][4];  ///< Interference angular coeffs S (i, k, n).
};

#endif /* DVCS_PROCESS_BMJ12_TORCH_H */
