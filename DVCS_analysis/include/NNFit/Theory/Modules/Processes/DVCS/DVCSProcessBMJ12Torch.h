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
 * For the tensor path it adds crossSectionTensorBatch(): the BMJ12
 * unpolarized-target cross section sigma(lambda, phi), batched over N
 * kinematic points x M phi nodes, with the CFFs taken as complex tensors from
 * DVCSCFFNNTorch so that the autograd graph runs from the NN parameters to
 * the cross section.
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

    // Sub-process cross sections sigma_X(lambda, phi), batched over N data
    // points x M phi nodes (tensor siblings of CrossSectionBH / CrossSectionVCS
    // / CrossSectionInterf). Each assumes the phi-independent setup has run;
    // the base crossSectionTensorBatch() template method drives setup + the
    // selected sum.
    torch::Tensor crossSectionBHTensorBatch(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) override;
    torch::Tensor crossSectionVCSTensorBatch(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) override;
    torch::Tensor crossSectionInterfTensorBatch(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) override;

protected:

    DVCSProcessBMJ12Torch(const DVCSProcessBMJ12Torch& other);

private:

    /**
     * Batched (N-point) sibling of setupKinematicsTorch: the BMJ12 derived
     * quantities and angular coefficients as [N]-tensor arithmetic, plus one
     * batched NN forward for the CFFs. Called once by the base
     * crossSectionTensorBatch() template method.
     */
    void setupKinematicsTorchBatch(const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E) override;

    // ----- batched tensor CFF layer -----------------------------------------

    /** Batched CFF tensor for GPD index F (0=H,1=E,2=Ht,3=Et), [N] complex. */
    torch::Tensor cffTensorBatch(int F, int a, int b) const;

    /** Batched siblings of C_VCS0(): [N] complex. */
    torch::Tensor C_VCS0Batch(int a1, int b1, int a2, int b2) const;
    torch::Tensor C_VCS0Batch(int a1, int b1, int a2, int b2, int a3, int b3) const;
    torch::Tensor C_VCS0Batch(int a1, int b1, int a2, int b2, int a3, int b3,
            int a4, int b4) const;

    /** Batched siblings of C_I0()/C_I0n()/S_I0n(): [N] complex. */
    torch::Tensor C_I0Batch(int a, int b, const std::string& VA) const;
    torch::Tensor C_I0nBatch(unsigned int n, int a, int b) const;
    torch::Tensor S_I0nBatch(unsigned int n, int a, int b) const;

    // ----- cached state ------------------------------------------------------

    /// Proton mass, M^2 -- a scalar constant, shared (unsuffixed) by the
    /// batched setup and CFF layer below; not per-point, so not tensorized.
    double m_M[2];

    // ----- batched cached state ----------------------------------------------
    // CFFs from the NN, batched ([N] complex double, grad-tracked)
    torch::Tensor m_CFFstdBatch[4];
    torch::Tensor m_CFFBatch[4][3];

    // BMJ12 kinematics, batched ([N] tensors, no gradient)
    torch::Tensor m_xBBatch, m_tBatch, m_Q2Batch;     ///< Cached raw kinematics.
    torch::Tensor m_xB2Batch;
    torch::Tensor m_QpowBatch[4];
    torch::Tensor m_Delta2Batch[2];
    torch::Tensor m_xBtQ2Batch[3];
    torch::Tensor m_yBMJBatch[3];
    torch::Tensor m_epsilonBMJBatch[2];
    torch::Tensor m_epsrootBatch[6];
    torch::Tensor m_KBatch[2];
    torch::Tensor m_KtBatch[2];
    torch::Tensor m_tminBMJBatch;
    torch::Tensor m_tmaxBMJBatch;
    torch::Tensor m_F1Batch, m_F2Batch;
    torch::Tensor m_phaseSpaceBMJBatch;
    torch::Tensor m_JBatch;
    torch::Tensor m_yepsBatch;
    torch::Tensor m_cFBatch[3];
    torch::Tensor m_cBH0Batch[3];
    torch::Tensor m_CangBatch[3][3][4];
    torch::Tensor m_SangBatch[3][3][4];
};

#endif /* DVCS_PROCESS_BMJ12_TORCH_H */
