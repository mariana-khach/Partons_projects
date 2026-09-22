//
// Created by Mariana Khachatryan on 6/15/26.
//
// Tensor twin of PARTONS::DVCSProcessBMJ12. The pure-kinematic BMJ12 machinery
// is transcribed verbatim from DVCSProcessBMJ12.cpp (doubles, no gradient); only
// the CFF-bilinear (VCS) and CFF-linear (interference) layers are evaluated in
// libtorch tensors, so the autograd graph runs from the NN CFFs to the cross
// section. Unpolarized target only (Lambda = 0).
//

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>

#include <cmath>

// Complete type needed as the SOURCE of the cross-cast below:
// DVCSProcessModule.h only forward-declares DVCSConvolCoeffFunctionModule.
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>

#include <partons/beans/Scales.h>
#include <partons/modules/scales/DVCS/DVCSScalesModule.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterModule.h>
#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFModuleTorch.h"

// ---------------------------------------------------------------------------
// Registration / boilerplate
// ---------------------------------------------------------------------------

const unsigned int DVCSProcessBMJ12Torch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSProcessBMJ12Torch("DVCSProcessBMJ12Torch"));

DVCSProcessBMJ12Torch::DVCSProcessBMJ12Torch(const std::string& className) :
        PARTONS::DVCSProcessBMJ12(className) {
}

DVCSProcessBMJ12Torch::DVCSProcessBMJ12Torch(
        const DVCSProcessBMJ12Torch& other) :
        PARTONS::DVCSProcessBMJ12(other) {
}

DVCSProcessBMJ12Torch::~DVCSProcessBMJ12Torch() {
}

DVCSProcessBMJ12Torch* DVCSProcessBMJ12Torch::clone() const {
    return new DVCSProcessBMJ12Torch(*this);
}

// ---------------------------------------------------------------------------
// Batched kinematics setup (phi-independent; [N]-tensor arithmetic, transcribed
// from DVCSProcessBMJ12.cpp -- same formulas as the scalar base, double -> tensor).
// ---------------------------------------------------------------------------

void DVCSProcessBMJ12Torch::setupKinematicsTorchBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2, const torch::Tensor& E) {

    m_xBBatch = xB;
    m_tBatch = t;
    m_Q2Batch = Q2;

    // Base-derived kinematics (DVCSProcessModule::initModule).
    const double M = PARTONS::Constant::PROTON_MASS;
    torch::Tensor epsilon = 2 * xB * M / torch::sqrt(Q2);
    torch::Tensor y = Q2 / (2 * xB * M * E);
    torch::Tensor eps2 = epsilon * epsilon;
    torch::Tensor epsrootBase = torch::sqrt(1 + eps2);
    torch::Tensor tfactor = -Q2 / (4 * xB * (1 - xB) + eps2);
    m_tminBMJBatch = tfactor * (2 * (1 - xB) * (1 - epsrootBase) + eps2);
    m_tmaxBMJBatch = tfactor * (2 * (1 - xB) * (1 + epsrootBase) + eps2);

    // BMJ12 derived kinematics (DVCSProcessBMJ12::initModule).
    m_xB2Batch = xB * xB;
    m_QpowBatch[0] = torch::sqrt(Q2);
    m_QpowBatch[1] = Q2;
    m_QpowBatch[2] = m_QpowBatch[0] * Q2;
    m_QpowBatch[3] = Q2 * Q2;
    m_Delta2Batch[0] = t;
    m_Delta2Batch[1] = t * t;
    m_xBtQ2Batch[0] = 2 - xB + xB * t / Q2;
    m_xBtQ2Batch[1] = m_xBtQ2Batch[0] * m_xBtQ2Batch[0];
    m_xBtQ2Batch[2] = m_xBtQ2Batch[1] * m_xBtQ2Batch[0];
    m_M[0] = M;
    m_M[1] = M * M;
    m_yBMJBatch[0] = y;
    m_yBMJBatch[1] = y * y;
    m_yBMJBatch[2] = y * y * y;
    m_epsilonBMJBatch[0] = epsilon;
    m_epsilonBMJBatch[1] = eps2;
    m_epsrootBatch[1] = 1 + eps2;
    m_epsrootBatch[0] = torch::sqrt(m_epsrootBatch[1]);
    m_epsrootBatch[2] = m_epsrootBatch[1] * m_epsrootBatch[0];
    m_epsrootBatch[3] = m_epsrootBatch[1] * m_epsrootBatch[1];
    m_epsrootBatch[4] = m_epsrootBatch[2] * m_epsrootBatch[1];
    m_epsrootBatch[5] = m_epsrootBatch[2] * m_epsrootBatch[2];
    m_KBatch[1] = -(t / Q2) * (1 - xB)
            * (1 - m_yBMJBatch[0] - m_yBMJBatch[1] * m_epsilonBMJBatch[1] / 4)
            * (1 - m_tminBMJBatch / t)
            * (m_epsrootBatch[0]
                    + (4 * xB * (1 - xB) + m_epsilonBMJBatch[1])
                            / (4 * (1 - xB)) * (t - m_tminBMJBatch) / Q2);
    m_KBatch[0] = torch::sqrt(m_KBatch[1]);
    m_KtBatch[1] = ((1 - xB) * xB + m_epsilonBMJBatch[1] / 4.)
            * (m_tminBMJBatch - t) * (t - m_tmaxBMJBatch) / Q2;
    m_KtBatch[0] = torch::sqrt(m_KtBatch[1]);

    // Dirac and Pauli form factors (DVCSProcessBMJ12::computeFormFactors).
    m_F1Batch = (4. * m_M[1] - 2.79285 * t)
            / (torch::pow(1. - 1.4084507042253522 * t, 2) * (4. * m_M[1] - 1. * t));
    m_F2Batch = (7.1714 * m_M[1])
            / (torch::pow(1 - 1.4084507042253522 * t, 2) * (4 * m_M[1] - t));

    // CFF coefficients cF[j][0] (DVCSProcessBMJ12::computeCFFs).
    for (int j = 0; j < 2; j++) {
        m_cFBatch[j] = (1. + (1 - 2 * j) * m_epsrootBatch[0]) / (2. * m_epsrootBatch[0])
                + (1 - xB) * m_xB2Batch * (4 * m_M[1] - t) * (1 + t / Q2)
                        / (Q2 * m_epsrootBatch[0] * m_xBtQ2Batch[1]);
    }
    {
        torch::Tensor A = -std::sqrt(2.) * m_KtBatch[0]
                / (m_epsrootBatch[0] * m_QpowBatch[0] * m_xBtQ2Batch[0]);
        m_cFBatch[2] = A * xB
                * (1 + 2. * xB * (4 * m_M[1] - t) / (Q2 * m_xBtQ2Batch[0]));
    }

    // Phase space.
    m_phaseSpaceBMJBatch = xB * m_yBMJBatch[1]
            / (1024 * std::pow(PARTONS::Constant::PI, 5) * m_QpowBatch[3]
                    * m_epsrootBatch[0]);

    // Lepton-propagator pieces (phi-independent part).
    torch::Tensor Delta2Q2 = t / Q2;
    m_yepsBatch = m_yBMJBatch[0] * (1. + m_epsilonBMJBatch[1]);
    m_JBatch = (1. - m_yBMJBatch[0] - m_yBMJBatch[0] * m_epsilonBMJBatch[1] / 2.)
                    * (1. + Delta2Q2)
            - (1. - xB) * (2. - m_yBMJBatch[0]) * Delta2Q2;

    // ----- Unpolarized BH Fourier coefficients (computeFourierCoeffsBH) -----
    {
        torch::Tensor F1 = m_F1Batch, F2 = m_F2Batch;
        torch::Tensor F22 = torch::pow(F2, 2);
        torch::Tensor F1PlusF2 = F1 + F2;
        torch::Tensor F1PlusF22 = torch::pow(F1PlusF2, 2);
        torch::Tensor Delta2M2 = t / (4 * m_M[1]);
        torch::Tensor F1MinusDeltaF2 = torch::pow(F1, 2) - Delta2M2 * F22;

        m_cBH0Batch[0] = 8. * m_KBatch[1]
                * ((2. + 3. * m_epsilonBMJBatch[1]) * F1MinusDeltaF2 / Delta2Q2
                        + 2. * m_xB2Batch * F1PlusF22)
                + torch::pow((2. - m_yBMJBatch[0]), 2)
                        * ((2. + m_epsilonBMJBatch[1])
                                * ((m_xB2Batch / Delta2M2)
                                                * torch::pow((1. + Delta2Q2), 2)
                                        + 4. * (1. - xB) * (1. + xB * Delta2Q2))
                                * F1MinusDeltaF2
                                + (4. * m_xB2Batch)
                                        * (xB
                                                + (1. - xB + m_epsilonBMJBatch[1] / 2.)
                                                        * torch::pow((1. - Delta2Q2), 2)
                                                - xB * (1. - 2. * xB)
                                                        * m_Delta2Batch[1] / m_QpowBatch[3])
                                        * F1PlusF22)
                + 8. * (1. + m_epsilonBMJBatch[1])
                        * (1. - m_yBMJBatch[0]
                                - m_epsilonBMJBatch[1] * m_yBMJBatch[1] / 4.)
                        * (2. * m_epsilonBMJBatch[1] * (1. - Delta2M2) * F1MinusDeltaF2
                                - m_xB2Batch * torch::pow((1. - Delta2Q2), 2) * F1PlusF22);
        m_cBH0Batch[1] = 8. * m_KBatch[0] * (2. - m_yBMJBatch[0])
                * ((m_xB2Batch / Delta2M2 - 2. * xB - m_epsilonBMJBatch[1])
                                * F1MinusDeltaF2
                        + 2. * m_xB2Batch * (1. - (1. - 2. * xB) * Delta2Q2)
                                * F1PlusF22);
        m_cBH0Batch[2] = 8. * m_xB2Batch * m_KBatch[1]
                * (F1MinusDeltaF2 / Delta2M2 + 2. * F1PlusF22);
    }

    // ----- Interference angular coefficients (computeAngularCoeffsInterf) ----
    // Only the S = 0 blocks (m_C and m_S) are needed for the unpolarized target.
    // Zero-fill every slot first (not every (i,k,n) combination is assigned
    // below, and a default-constructed torch::Tensor is undefined, not zero,
    // unlike a plain double -- see the m_Cang/m_Sang doc comment history).
    {
        torch::Tensor zero = torch::zeros_like(xB);
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) {
                for (int n = 0; n < 4; ++n) {
                    m_CangBatch[i][k][n] = zero;
                    m_SangBatch[i][k][n] = zero;
                }
            }
        }
    }
    {
        torch::Tensor tQ2 = t / Q2;
        torch::Tensor Kt2Q2 = m_KtBatch[1] / Q2;
        torch::Tensor tp = t - m_tminBMJBatch;
        torch::Tensor tpQ2 = tp / Q2;
        torch::Tensor C2 = 1. - m_yBMJBatch[0]
                - m_epsilonBMJBatch[1] * m_yBMJBatch[1] / 4.;
        torch::Tensor C1 = torch::sqrt(C2);
        torch::Tensor B2 = 2. - 2 * m_yBMJBatch[0] + m_yBMJBatch[1]
                + m_epsilonBMJBatch[1] * m_yBMJBatch[1] / 2.;

        torch::Tensor (*m_C)[3][4] = m_CangBatch; // local aliases matching the source names
        torch::Tensor (*m_S)[3][4] = m_SangBatch;

        // C++ coefficients
        m_C[0][0][0] = -4 * (2. - m_yBMJBatch[0]) * (1. + m_epsrootBatch[0])
                / m_epsrootBatch[3]
                * (Kt2Q2 * (2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                                / m_epsrootBatch[0]
                        + tQ2 * C2 * (2 - xB)
                                * (1.
                                        + (2 * xB
                                                * (2 - xB + (m_epsrootBatch[0] - 1.) / 2.
                                                        + m_epsilonBMJBatch[1] / (2 * xB))
                                                * tQ2 + m_epsilonBMJBatch[1])
                                                / ((2. - xB) * (1 + m_epsrootBatch[0]))));
        m_C[0][1][0] = 8 * (2. - m_yBMJBatch[0]) * xB * tQ2 / m_epsrootBatch[3]
                * (Kt2Q2 * (2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                                / m_epsrootBatch[0]
                        + C2 * (1 + m_epsrootBatch[0]) / 2. * (1. + tQ2)
                                * (1.
                                        + (m_epsrootBatch[0] - 1. + 2 * xB) * tQ2
                                                / (1 + m_epsrootBatch[0])));
        m_C[0][2][0] = 8 * (2. - m_yBMJBatch[0]) * tQ2 / m_epsrootBatch[3]
                * (Kt2Q2 * (2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                                / m_epsrootBatch[0]
                        * (1 + m_epsrootBatch[0] - 2 * xB) / 2.
                        - C2
                                * (2 * Kt2Q2
                                        - (1 + m_epsrootBatch[0]) / 2.
                                                * (1 + m_epsrootBatch[0] - xB
                                                        + (m_epsrootBatch[0] - 1.
                                                                + xB * (3. + m_epsrootBatch[0]
                                                                        - 2 * xB)
                                                                        / (1 + m_epsrootBatch[0]))
                                                                * tQ2)));
        m_C[0][0][1] = -16 * m_KBatch[0] * C2 / m_epsrootBatch[4]
                * ((1. + (1. - xB) * (m_epsrootBatch[0] - 1.) / (2 * xB)
                        + m_epsilonBMJBatch[1] / (4 * xB)) * xB * tQ2
                        - 3 * m_epsilonBMJBatch[1] / 4.)
                - 4 * m_KBatch[0] * B2
                        * (1. + m_epsrootBatch[0] - m_epsilonBMJBatch[1]) / m_epsrootBatch[4]
                        * (1. - (1. - 3 * xB) * tQ2
                                + (1. - m_epsrootBatch[0] + 3 * m_epsilonBMJBatch[1]) * xB * tQ2
                                        / (1. + m_epsrootBatch[0] - m_epsilonBMJBatch[1]));
        m_C[0][1][1] = 16 * m_KBatch[0] * xB * tQ2 / m_epsrootBatch[4]
                * ((2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                                * (1. - (1. - 2 * xB) * tQ2)
                        + C2 * (1 + m_epsrootBatch[0] - 2 * xB) * tpQ2 / 2.);
        m_C[0][2][1] = -16 * m_KBatch[0] * tQ2 / m_epsrootBatch[3]
                * (C2
                        * (1. - (1. - 2 * xB) * tQ2
                                + (4 * xB * (1. - xB) + m_epsilonBMJBatch[1]) * tpQ2
                                        / (4 * m_epsrootBatch[0]))
                        - (2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                                * (1. - xB / 2.
                                        + (1 + m_epsrootBatch[0] - 2 * xB) / 4. * (1. - tQ2)
                                        + (4 * xB * (1. - xB) + m_epsilonBMJBatch[1])
                                                * tpQ2 / (2 * m_epsrootBatch[0])));
        m_C[0][0][2] = 8 * (2. - m_yBMJBatch[0]) * C2 / m_epsrootBatch[3]
                * (2 * m_epsilonBMJBatch[1] * Kt2Q2 / (m_epsrootBatch[0] + m_epsrootBatch[1])
                        + xB * tQ2 * tpQ2
                                * (1. - xB - (m_epsrootBatch[0] - 1.) / 2.
                                        + m_epsilonBMJBatch[1] / (2 * xB)));
        m_C[0][1][2] = 8 * (2. - m_yBMJBatch[0]) * C2 * xB * tQ2 / m_epsrootBatch[3]
                * (4 * Kt2Q2 / m_epsrootBatch[0]
                        + (1. + m_epsrootBatch[0] - 2 * xB) / 2. * (1 + tQ2) * tpQ2);
        m_C[0][2][2] = 4 * (2. - m_yBMJBatch[0]) * C2 * tQ2 / m_epsrootBatch[3]
                * (4 * (1. - 2 * xB) * Kt2Q2 / m_epsrootBatch[0]
                        - (3. - m_epsrootBatch[0] - 2 * xB + m_epsilonBMJBatch[1] / xB)
                                * xB * tpQ2);
        m_C[0][0][3] = -8 * m_KBatch[0] * C2 * (m_epsrootBatch[0] - 1.) / m_epsrootBatch[4]
                * ((1. - xB) * tQ2 + (m_epsrootBatch[0] - 1.) / 2. * (1 + tQ2));
        m_C[0][1][3] = -8 * m_KBatch[0] * C2 * xB * tQ2 / m_epsrootBatch[4]
                * (m_epsrootBatch[0] - 1. + (1. + m_epsrootBatch[0] - 2 * xB) * tQ2);
        m_C[0][2][3] = 16 * m_KBatch[0] * C2 * tQ2 * tpQ2 / m_epsrootBatch[4]
                * (xB * (1. - xB) + m_epsilonBMJBatch[1] / 4.);

        // S++ coefficients
        m_S[0][0][1] = 8 * m_KBatch[0] * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0]
                / m_epsrootBatch[1]
                * (1. + (1. - xB + (m_epsrootBatch[0] - 1.) / 2.) / m_epsrootBatch[1] * tpQ2);
        m_S[0][1][1] = -8 * m_KBatch[0] * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0] * xB * tQ2
                / m_epsrootBatch[3]
                * (m_epsrootBatch[0] - 1. + (1. + m_epsrootBatch[0] - 2 * xB) * tQ2);
        m_S[0][2][1] = 8 * m_KBatch[0] * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0] * tQ2
                / m_epsrootBatch[1]
                * (1.
                        - (1. - 2 * xB) * (1. + m_epsrootBatch[0] - 2 * xB)
                                / (2 * m_epsrootBatch[1]) * tpQ2);
        m_S[0][0][2] = -4 * C2 * m_yBMJBatch[0] / m_epsrootBatch[2]
                * (1. + m_epsrootBatch[0] - 2 * xB) * tpQ2
                * ((m_epsilonBMJBatch[1] - xB * (m_epsrootBatch[0] - 1.))
                        / (1. + m_epsrootBatch[0] - 2 * xB)
                        - (2 * xB + m_epsilonBMJBatch[1]) * tpQ2 / (2 * m_epsrootBatch[0]));
        m_S[0][1][2] = -4 * C2 * m_yBMJBatch[0] * xB * tQ2 / m_epsrootBatch[3]
                * (1. - (1. - 2 * xB) * tQ2)
                * (m_epsrootBatch[0] - 1. + (1. + m_epsrootBatch[0] - 2 * xB) * tQ2);
        m_S[0][2][2] = -8 * C2 * m_yBMJBatch[0] * tQ2 * tpQ2 / m_epsrootBatch[3]
                * (1. - xB / 2. + 3 * m_epsilonBMJBatch[1] / 4.)
                * (1. + m_epsrootBatch[0] - 2 * xB)
                * (1.
                        + (4 * (1 - xB) * xB + m_epsilonBMJBatch[1])
                                / (4. - 2 * xB + 3 * m_epsilonBMJBatch[1]) * tQ2);

        // C0+
        m_C[2][0][0] = 12 * std::sqrt(2.) * m_KBatch[0] * (2. - m_yBMJBatch[0]) * C1
                / m_epsrootBatch[4]
                * (m_epsilonBMJBatch[1] + (2. - 6 * xB - m_epsilonBMJBatch[1]) / 3. * tQ2);
        m_C[2][1][0] = 24 * std::sqrt(2.) * m_KBatch[0] * (2. - m_yBMJBatch[0]) * C1 * xB
                * tQ2 / m_epsrootBatch[4] * (1. - (1. - 2 * xB) * tQ2);
        m_C[2][2][0] = 4 * std::sqrt(2.) * m_KBatch[0] * (2. - m_yBMJBatch[0]) * C1 * tQ2
                / m_epsrootBatch[4] * (8. - 6 * xB + 5 * m_epsilonBMJBatch[1])
                * (1.
                        - tQ2 * (2. - 12 * xB * (1. - xB) - m_epsilonBMJBatch[1])
                                / (8. - 6 * xB + 5 * m_epsilonBMJBatch[1]));
        m_C[2][0][1] = 8 * std::sqrt(2.) * C1 / m_epsrootBatch[3]
                * ((2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0]) * tpQ2
                        * (1. - xB
                                + ((1. - xB) * xB + m_epsilonBMJBatch[1] / 4.)
                                        / m_epsrootBatch[0] * tpQ2)
                        + C2 / m_epsrootBatch[0] * (1. - (1. - 2 * xB) * tQ2)
                                * (m_epsilonBMJBatch[1]
                                        - 2 * (1. + m_epsilonBMJBatch[1] / (2 * xB))
                                                * xB * tQ2));
        m_C[2][1][1] = 16 * std::sqrt(2.) * C1 * xB * tQ2 / m_epsrootBatch[4]
                * (Kt2Q2 * (2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                        + torch::pow(1. - (1. - 2 * xB) * tQ2, 2) * C2);
        m_C[2][2][1] = 8 * std::sqrt(2.) * C1 * tQ2 / m_epsrootBatch[4]
                * (Kt2Q2 * (1. - 2 * xB) * (2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                        + (1. - (1. - 2 * xB) * tQ2) * C2
                                * (4. - 2 * xB + 3 * m_epsilonBMJBatch[1]
                                        + tQ2 * (4 * xB * (1. - xB) + m_epsilonBMJBatch[1])));
        m_C[2][0][2] = -8 * std::sqrt(2.) * m_KBatch[0] * (2. - m_yBMJBatch[0]) * C1
                / m_epsrootBatch[4]
                * (1. + m_epsilonBMJBatch[1] / 2.)
                * (1.
                        + (1. + m_epsilonBMJBatch[1] / (2 * xB))
                                / (1. + m_epsilonBMJBatch[1] / 2.) * xB * tQ2);
        m_C[2][1][2] = 8 * std::sqrt(2.) * m_KBatch[0] * (2. - m_yBMJBatch[0]) * C1 * xB
                * tQ2 / m_epsrootBatch[4] * (1. - (1. - 2 * xB) * tQ2);
        m_C[2][2][2] = 8 * std::sqrt(2.) * m_KBatch[0] * (2. - m_yBMJBatch[0]) * C1 * tQ2
                / m_epsrootBatch[3]
                * (1. - xB
                        + tpQ2 / 2. * (4 * xB * (1. - xB) + m_epsilonBMJBatch[1])
                                / m_epsrootBatch[0]);

        // S0+
        m_S[2][0][1] = 8 * std::sqrt(2.) * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0] * C1
                * Kt2Q2 / m_epsrootBatch[3];
        m_S[2][1][1] = 4 * std::sqrt(2.) * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0] * C1 * xB
                * tQ2 / m_epsrootBatch[3]
                * (4 * (1. - xB) * tQ2 * (1 + xB * tQ2)
                        + m_epsilonBMJBatch[1] * (1 + tQ2) * (1 + tQ2));
        m_S[2][2][1] = -8 * std::sqrt(2.) * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0]
                * (1. - 2 * xB) * C1 * tQ2 * Kt2Q2 / m_epsrootBatch[3];
        m_S[2][0][2] = 8 * std::sqrt(2.) * m_KBatch[0] * m_yBMJBatch[0] * C1
                * (1. + m_epsilonBMJBatch[1] / 2.) / m_epsrootBatch[3]
                * (1.
                        + (1. + m_epsilonBMJBatch[1] / (2 * xB))
                                / (1. + m_epsilonBMJBatch[1] / 2.) * xB * tQ2);
        m_S[2][1][2] = -8 * std::sqrt(2.) * m_KBatch[0] * m_yBMJBatch[0] * C1 * xB * tQ2
                / m_epsrootBatch[3] * (1. - (1. - 2 * xB) * tQ2);
        m_S[2][2][2] = -2 * std::sqrt(2.) * m_KBatch[0] * m_yBMJBatch[0] * C1 * tQ2
                / m_epsrootBatch[3]
                * (4. - 4 * xB + 2 * m_epsilonBMJBatch[1]
                        + 2 * tQ2 * (4 * xB * (1. - xB) + m_epsilonBMJBatch[1]));

        // C-+
        m_C[1][0][0] = 8 * (2. - m_yBMJBatch[0]) / m_epsrootBatch[2]
                * ((2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0]) * (m_epsrootBatch[0] - 1.)
                                / (2 * m_epsrootBatch[1]) * Kt2Q2
                        + C2 / m_epsrootBatch[0]
                                * (1. - xB - (m_epsrootBatch[0] - 1.) / 2.
                                        + m_epsilonBMJBatch[1] / (2 * xB)) * xB * tQ2
                                * tpQ2);
        m_C[1][1][0] = 4 * (2. - m_yBMJBatch[0]) * xB * tQ2 / m_epsrootBatch[4]
                * (2 * Kt2Q2 * B2
                        - (1. - (1. - 2 * xB) * tQ2) * C2
                                * (m_epsrootBatch[0] - 1.
                                        + (m_epsrootBatch[0] + 1. - 2 * xB) * tQ2));
        m_C[1][2][0] = 4 * (2. - m_yBMJBatch[0]) * tQ2 / m_epsrootBatch[3]
                * (tpQ2 * C2
                        * (2 * m_xB2Batch - m_epsilonBMJBatch[1] - 3 * xB
                                + xB * m_epsrootBatch[0])
                        + Kt2Q2 / m_epsrootBatch[0]
                                * (4.
                                        - 2 * xB * (2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0])
                                        - 4 * m_yBMJBatch[0]
                                        + m_yBMJBatch[1] - m_yBMJBatch[1] * m_epsrootBatch[2]));
        m_C[1][0][1] = 8 * m_KBatch[0] / m_epsrootBatch[2]
                * ((2. - m_yBMJBatch[0]) * (2. - m_yBMJBatch[0]) * (2. - m_epsrootBatch[0])
                                / m_epsrootBatch[1]
                        * ((m_epsrootBatch[0] - 1. + m_epsilonBMJBatch[1])
                                        / (2 * (2. - m_epsrootBatch[0]))
                                        * (1. - tQ2) - xB * tQ2)
                        + 2 * C2 / m_epsrootBatch[0]
                                * ((1. - m_epsrootBatch[0] + m_epsilonBMJBatch[1] / 2.)
                                                / (2 * m_epsrootBatch[0])
                                        + tQ2
                                                * (1. - 3 * xB / 2.
                                                        + (xB + m_epsilonBMJBatch[1] / 2.)
                                                                / (2 * m_epsrootBatch[0]))));
        m_C[1][1][1] = 8 * m_KBatch[0] * xB * tQ2 / m_epsrootBatch[4]
                * (2 * (1. - (1. - 2 * xB) * tQ2) * B2
                        + C2
                                * (3. - m_epsrootBatch[0]
                                        - (3 * (1. - 2 * xB) + m_epsrootBatch[0]) * tQ2));
        m_C[1][2][1] = 4 * m_KBatch[0] * tQ2 / m_epsrootBatch[4]
                * (B2
                        * (5. - 4 * xB + 3 * m_epsilonBMJBatch[1] - m_epsrootBatch[0]
                                - tQ2
                                        * (1. - m_epsilonBMJBatch[1] - m_epsrootBatch[0]
                                                - 2 * xB
                                                        * (4. - 4 * xB - m_epsrootBatch[0])))
                        + C2
                                * (8. + 5 * m_epsilonBMJBatch[1] - 6 * xB
                                        + 2 * xB * m_epsrootBatch[0]
                                        - tQ2
                                                * (2. - m_epsilonBMJBatch[1]
                                                        + 2 * m_epsrootBatch[0]
                                                        - 4 * xB
                                                                * (3. - 3 * xB
                                                                        + m_epsrootBatch[0]))));
        m_C[1][0][2] = 4 * (2. - m_yBMJBatch[0]) * C2 * (1. + m_epsrootBatch[0])
                / m_epsrootBatch[4]
                * ((2. - 3 * xB) * tQ2
                        + (1. - 2 * xB + 2 * (1. - xB) / (1. + m_epsrootBatch[0]))
                                * xB * tQ2 * tQ2
                        + (1.
                                + (m_epsrootBatch[0] + xB + (1. - xB) * tQ2)
                                        / (1. + m_epsrootBatch[0]) * tQ2)
                                * m_epsilonBMJBatch[1]);
        m_C[1][1][2] = 4 * (2. - m_yBMJBatch[0]) * C2 * xB * tQ2 / m_epsrootBatch[4]
                * (4 * Kt2Q2 + 1. + m_epsrootBatch[0]
                        + tQ2
                                * ((1. - 2 * xB) * (1. - 2 * xB - m_epsrootBatch[0]) * tQ2
                                        - 2. + 4 * xB + 2 * xB * m_epsrootBatch[0]));
        m_C[1][2][2] = 16 * (2. - m_yBMJBatch[0]) * C2 * tQ2 / m_epsrootBatch[2]
                * (Kt2Q2 * (1. - 2 * xB) / m_epsrootBatch[1]
                        - (1. - xB) / (4 * xB * (1. - xB) + m_epsilonBMJBatch[1])
                                * (2 * m_xB2Batch - m_epsilonBMJBatch[1] - 3 * xB
                                        - xB * m_epsrootBatch[0])
                        - tpQ2
                                * (2 * m_xB2Batch - m_epsilonBMJBatch[1] - 3 * xB
                                        - xB * m_epsrootBatch[0]) / (4 * m_epsrootBatch[0]));
        m_C[1][0][3] = -8 * m_KBatch[0] * C2
                * (1. + m_epsrootBatch[0] + m_epsilonBMJBatch[1] / 2.) / m_epsrootBatch[4]
                * (1.
                        + (1. + m_epsrootBatch[0] + m_epsilonBMJBatch[1] / (2 * xB))
                                / (1. + m_epsrootBatch[0] + m_epsilonBMJBatch[1] / 2)
                                * xB * tQ2);
        m_C[1][1][3] = 8 * m_KBatch[0] * C2 * xB * tQ2 / m_epsrootBatch[4]
                * (1. + m_epsrootBatch[0])
                * (1. - tQ2 * (1. - 2 * xB - m_epsrootBatch[0]) / (1. + m_epsrootBatch[0]));
        m_C[1][2][3] = 16 * m_KBatch[0] * C2 * tQ2 / m_epsrootBatch[3]
                * (1. - xB
                        + tpQ2 * (4 * xB * (1. - xB) + m_epsilonBMJBatch[1])
                                / (4 * m_epsrootBatch[0]));

        // S-+
        m_S[1][0][1] = 4 * m_KBatch[0] * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0]
                / m_epsrootBatch[3]
                * (1. - m_epsrootBatch[0] + 2 * m_epsilonBMJBatch[1]
                        - 2 * (1. + (m_epsrootBatch[0] - 1.) / (2 * xB)) * xB * tQ2);
        m_S[1][1][1] = 8 * m_KBatch[0] * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0] * xB * tQ2
                / m_epsrootBatch[3] * (1. + m_epsrootBatch[0])
                * (1. - tQ2 * (1. - 2 * xB - m_epsrootBatch[0]) / (1. + m_epsrootBatch[0]));
        m_S[1][2][1] = 4 * m_KBatch[0] * (2. - m_yBMJBatch[0]) * m_yBMJBatch[0] * tQ2
                / m_epsrootBatch[3]
                * (3. + 2 * m_epsilonBMJBatch[1] + m_epsrootBatch[0]
                        - 2 * xB * (1. + m_epsrootBatch[0])
                        - tQ2 * (1. - 2 * xB) * (1. - 2 * xB - m_epsrootBatch[0]));
        m_S[1][0][2] = 2 * m_yBMJBatch[0] * C2 * (1. + m_epsrootBatch[0]) / m_epsrootBatch[3]
                * (m_epsilonBMJBatch[1]
                        - 2 * (1. + m_epsilonBMJBatch[1] / (2 * xB)) * xB * tQ2)
                * (1. + (m_epsrootBatch[0] - 1. + 2 * xB) / (1. + m_epsrootBatch[0]) * tQ2);
        m_S[1][1][2] = 4 * m_yBMJBatch[0] * C2 * xB * tQ2 / m_epsrootBatch[3]
                * (1. + m_epsrootBatch[0]) * (1. - (1. - 2 * xB) * tQ2)
                * (1. - tQ2 * (1. - 2 * xB - m_epsrootBatch[0]) / (1. + m_epsrootBatch[0]));
        m_S[1][2][2] = 2 * m_yBMJBatch[0] * C2 * tQ2 / m_epsrootBatch[3]
                * (4. - 2 * xB + 3 * m_epsilonBMJBatch[1]
                        + tQ2 * (4 * xB * (1. - xB) + m_epsilonBMJBatch[1]))
                * (1. + m_epsrootBatch[0] - tQ2 * (1. - 2 * xB - m_epsrootBatch[0]));
    }

    // ----- CFFs as tensors, batched ----------------------------------------
    // The process module converts, the CFF module receives -- exactly as in
    // DVCSProcessModule::computeConvolCoeffFunction, which runs the
    // xi-converter and the scales module and hands the CFF module
    // (xi, t, Q2, muF2, muR2). Same modules this process is wired with, so the
    // torch path cannot drift from the scalar path's conventions. N scalar
    // calls, once per prepare (not per phi node, not per epoch step).
    const int64_t nPts = xB.size(0);
    std::vector<double> xiVec(nPts), muF2Vec(nPts), muR2Vec(nPts);
    for (int64_t i = 0; i < nPts; ++i) {
        PARTONS::DVCSObservableKinematic kin(xB[i].item<double>(),
                t[i].item<double>(), Q2[i].item<double>(), E[i].item<double>(),
                0.); // phi is irrelevant to both modules
        xiVec[i]   = m_pXiConverterModule->compute(kin).getValue();
        PARTONS::Scales scale = m_pScaleModule->compute(kin);
        muF2Vec[i] = scale.getMuF2().getValue();
        muR2Vec[i] = scale.getMuR2().getValue();
    }
    const torch::TensorOptions f64opt = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor xiT   = torch::tensor(xiVec, f64opt);
    torch::Tensor muF2T = torch::tensor(muF2Vec, f64opt);
    torch::Tensor muR2T = torch::tensor(muR2Vec, f64opt);

    // Cross-cast to the tensor interface, not to a concrete module: any CFF
    // source implementing DVCSCFFModuleTorch can drive this chain -- the
    // trained network, or DVCSCFFScalarTorch wrapping a scalar PARTONS model.
    DVCSCFFModuleTorch* pCFF =
            dynamic_cast<DVCSCFFModuleTorch*>(m_pConvolCoeffFunctionModule);
    if (!pCFF) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSCFFModuleTorch convol-coeff module.");
    }
    DVCSCFFModuleTorch::AllCFFsTensorBatch cffs =
            pCFF->computeAllCFFsTensorBatch(xiT, t, Q2, muF2T, muR2T);
    m_CFFstdBatch[0] = cffs.H;
    m_CFFstdBatch[1] = cffs.E;
    m_CFFstdBatch[2] = cffs.Ht;
    m_CFFstdBatch[3] = cffs.Et;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            m_CFFBatch[i][j] = m_cFBatch[j] * m_CFFstdBatch[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Batched tensor CFF layer
// ---------------------------------------------------------------------------

torch::Tensor DVCSProcessBMJ12Torch::cffTensorBatch(int F, int a, int b) const {
    if (a == -1) {
        a = 1;
        b = -b;
    }
    int j;
    if (a == 1 && b == 1) {
        j = 0;
    } else if (a == 1 && b == -1) {
        j = 1;
    } else if (a == 0 && std::abs(b) == 1) {
        j = 2;
    } else {
        return torch::complex(torch::zeros_like(m_xBBatch), torch::zeros_like(m_xBBatch));
    }
    return m_CFFBatch[F][j];
}

torch::Tensor DVCSProcessBMJ12Torch::C_VCS0Batch(int a1, int b1, int a2, int b2) const {
    torch::Tensor H1 = cffTensorBatch(0, a1, b1), E1 = cffTensorBatch(1, a1, b1),
            Ht1 = cffTensorBatch(2, a1, b1), Et1 = cffTensorBatch(3, a1, b1);
    torch::Tensor H2 = cffTensorBatch(0, a2, b2), E2 = cffTensorBatch(1, a2, b2),
            Ht2 = cffTensorBatch(2, a2, b2), Et2 = cffTensorBatch(3, a2, b2);

    torch::Tensor tQ2 = m_tBatch / m_Q2Batch;
    return 4. * (1 - m_xBBatch) * (1 + m_xBBatch * tQ2) / m_xBtQ2Batch[1]
            * (H1 * torch::conj(H2) + Ht1 * torch::conj(Ht2))
            + (2 + tQ2) * m_epsilonBMJBatch[1] / m_xBtQ2Batch[1] * Ht1 * torch::conj(Ht2)
            - m_tBatch / (4 * m_M[1]) * E1 * torch::conj(E2)
            - m_xB2Batch / m_xBtQ2Batch[1]
                    * (torch::pow(1 + tQ2, 2)
                            * (H1 * torch::conj(E2) + E1 * torch::conj(H2)
                                    + E1 * torch::conj(E2))
                            + Ht1 * torch::conj(Et2) + Et1 * torch::conj(Ht2)
                            + m_tBatch / (4 * m_M[1]) * Et1 * torch::conj(Et2));
}

torch::Tensor DVCSProcessBMJ12Torch::C_VCS0Batch(int a1, int b1, int a2, int b2,
        int a3, int b3) const {
    // S = 0 -> signe = +1
    return C_VCS0Batch(a1, b1, a2, b2) + C_VCS0Batch(a1, b1, a3, b3);
}

torch::Tensor DVCSProcessBMJ12Torch::C_VCS0Batch(int a1, int b1, int a2, int b2,
        int a3, int b3, int a4, int b4) const {
    // S = 0 -> signe = +1
    return C_VCS0Batch(a1, b1, a2, b2) + C_VCS0Batch(a3, b3, a4, b4);
}

torch::Tensor DVCSProcessBMJ12Torch::C_I0Batch(int a, int b,
        const std::string& VA) const {
    torch::Tensor H = cffTensorBatch(0, a, b), E = cffTensorBatch(1, a, b),
            Ht = cffTensorBatch(2, a, b), Et = cffTensorBatch(3, a, b);

    torch::Tensor F1PlusF2 = m_F1Batch + m_F2Batch;
    torch::Tensor xBF1PlusF2 = m_xBBatch * F1PlusF2 / m_xBtQ2Batch[0];

    if (VA == "V") {
        return xBF1PlusF2 * (H + E);
    } else if (VA == "A") {
        return xBF1PlusF2 * Ht;
    }
    return m_F1Batch * H - m_tBatch / (4 * m_M[1]) * m_F2Batch * E + xBF1PlusF2 * Ht;
}

torch::Tensor DVCSProcessBMJ12Torch::C_I0nBatch(unsigned int n, int a, int b) const {
    if (a == -1) {
        a = 1;
        b = -b;
    }
    int i;
    if (a == 1 && b == 1) {
        i = 0;
    } else if (a == 1 && b == -1) {
        i = 1;
    } else if (a == 0 && std::abs(b) == 1) {
        i = 2;
    } else {
        return torch::complex(torch::zeros_like(m_xBBatch), torch::zeros_like(m_xBBatch));
    }
    if (n >= 4) {
        return torch::complex(torch::zeros_like(m_xBBatch), torch::zeros_like(m_xBBatch));
    }
    return m_CangBatch[i][0][n] * C_I0Batch(a, b, "")
            + m_CangBatch[i][1][n] * C_I0Batch(a, b, "V")
            + m_CangBatch[i][2][n] * C_I0Batch(a, b, "A");
}

torch::Tensor DVCSProcessBMJ12Torch::S_I0nBatch(unsigned int n, int a, int b) const {
    if (a == -1) {
        a = 1;
        b = -b;
    }
    int i;
    if (a == 1 && b == 1) {
        i = 0;
    } else if (a == 1 && b == -1) {
        i = 1;
    } else if (a == 0 && std::abs(b) == 1) {
        i = 2;
    } else {
        return torch::complex(torch::zeros_like(m_xBBatch), torch::zeros_like(m_xBBatch));
    }
    if (n >= 4) {
        return torch::complex(torch::zeros_like(m_xBBatch), torch::zeros_like(m_xBBatch));
    }
    return m_SangBatch[i][0][n] * C_I0Batch(a, b, "")
            + m_SangBatch[i][1][n] * C_I0Batch(a, b, "V")
            + m_SangBatch[i][2][n] * C_I0Batch(a, b, "A");
}

// ---------------------------------------------------------------------------
// Batched cross section sigma(lambda, phi), [N,M] (N data points x M phi nodes)
// ---------------------------------------------------------------------------

torch::Tensor DVCSProcessBMJ12Torch::crossSectionBHTensorBatch(double beamHelicity,
        double beamCharge, const torch::Tensor& phi) {

    (void) beamHelicity;
    (void) beamCharge;

    auto bc = [](const torch::Tensor& x) { return x.unsqueeze(1); }; // [N] -> [N,1]

    const double e6 = std::pow(PARTONS::Constant::POSITRON_CHARGE, 6);

    torch::Tensor phi1 = PARTONS::Constant::PI - phi; // [M]
    auto cosn = [&](int n) { return torch::cos(double(n) * phi1); }; // [M]

    torch::Tensor P1 = -(bc(m_JBatch) + 2 * bc(m_KBatch[0]) * torch::cos(phi1))
            / bc(m_yepsBatch); // [N,M]
    torch::Tensor P2 = (1. + bc(m_tBatch) / bc(m_Q2Batch)) - P1; // [N,M]

    torch::Tensor sqrBH = bc(m_cBH0Batch[0]) * cosn(0) + bc(m_cBH0Batch[1]) * cosn(1)
            + bc(m_cBH0Batch[2]) * cosn(2); // [N,M]
    torch::Tensor denomConst =
            bc(m_xB2Batch * m_yBMJBatch[1] * m_epsrootBatch[3] * m_tBatch); // [N,1]
    torch::Tensor A_BH = e6 / (denomConst * P1 * P2); // [N,M]
    return bc(m_phaseSpaceBMJBatch) * A_BH * sqrBH; // [N,M]
}

torch::Tensor DVCSProcessBMJ12Torch::crossSectionVCSTensorBatch(double beamHelicity,
        double beamCharge, const torch::Tensor& phi) {

    (void) beamCharge; // VCS is independent of beam charge.

    auto bc = [](const torch::Tensor& x) { return x.unsqueeze(1); }; // [N] -> [N,1]

    const double lambda = beamHelicity;
    const double e6 = std::pow(PARTONS::Constant::POSITRON_CHARGE, 6);

    // ----- VCS unpolarized Fourier coefficients ([N] real tensors) ---------
    torch::Tensor C2 = 1. - m_yBMJBatch[0]
            - m_epsilonBMJBatch[1] * m_yBMJBatch[1] / 4.;
    torch::Tensor C1 = torch::sqrt(C2);
    torch::Tensor cVCS0 = (2 * (2 - 2 * m_yBMJBatch[0]
            + m_yBMJBatch[1] * (1. + m_epsilonBMJBatch[1] / 2.))
            * torch::real(C_VCS0Batch(1, 1, 1, 1, -1, 1, -1, 1))
            + 8 * C2 * torch::real(C_VCS0Batch(0, 1, 0, 1))) / m_epsrootBatch[1];
    torch::Tensor cVCS1 = 4 * std::sqrt(2.) * C1 / m_epsrootBatch[1] * (2 - m_yBMJBatch[0])
            * torch::real(C_VCS0Batch(0, 1, 1, 1, -1, 1));
    torch::Tensor sVCS1 = 4 * std::sqrt(2.) * C1 / m_epsrootBatch[1]
            * (-lambda * m_yBMJBatch[0] * m_epsrootBatch[0])
            * torch::imag(C_VCS0Batch(0, 1, 1, 1, -1, 1));
    torch::Tensor cVCS2 = 8 * C2 / m_epsrootBatch[1] * torch::real(C_VCS0Batch(-1, 1, 1, 1));

    torch::Tensor phi1 = PARTONS::Constant::PI - phi; // [M]
    auto cosn = [&](int n) { return torch::cos(double(n) * phi1); };
    auto sinn = [&](int n) { return torch::sin(double(n) * phi1); };

    torch::Tensor A_VCS = e6 / (m_yBMJBatch[1] * m_Q2Batch); // [N]
    torch::Tensor sqrVCS = bc(cVCS0) * cosn(0) + bc(cVCS1) * cosn(1)
            + bc(cVCS2) * cosn(2) + bc(sVCS1) * sinn(1); // [N,M]
    return bc(m_phaseSpaceBMJBatch) * bc(A_VCS) * sqrVCS; // [N,M]
}

torch::Tensor DVCSProcessBMJ12Torch::crossSectionInterfTensorBatch(
        double beamHelicity, double beamCharge, const torch::Tensor& phi) {

    auto bc = [](const torch::Tensor& x) { return x.unsqueeze(1); }; // [N] -> [N,1]

    const double lambda = beamHelicity;
    const double e6 = std::pow(PARTONS::Constant::POSITRON_CHARGE, 6);

    // ----- Interference unpolarized Fourier coefficients ([N] real) --------
    torch::Tensor cI[4];
    for (unsigned int n = 0; n < 4; n++) {
        cI[n] = torch::real(C_I0nBatch(n, 1, 1) + C_I0nBatch(n, -1, 1)
                + C_I0nBatch(n, 0, 1));
    }
    torch::Tensor zeroN = torch::zeros_like(m_xBBatch);
    torch::Tensor sI[4] = { zeroN, zeroN, zeroN, zeroN };
    for (unsigned int n = 1; n < 3; n++) {
        sI[n] = lambda
                * torch::imag(S_I0nBatch(n, 1, 1) + S_I0nBatch(n, -1, 1)
                        + S_I0nBatch(n, 0, 1));
    }

    torch::Tensor phi1 = PARTONS::Constant::PI - phi; // [M]
    auto cosn = [&](int n) { return torch::cos(double(n) * phi1); };
    auto sinn = [&](int n) { return torch::sin(double(n) * phi1); };

    torch::Tensor P1 = -(bc(m_JBatch) + 2 * bc(m_KBatch[0]) * torch::cos(phi1))
            / bc(m_yepsBatch); // [N,M]
    torch::Tensor P2 = (1. + bc(m_tBatch) / bc(m_Q2Batch)) - P1; // [N,M]

    torch::Tensor denomConst = bc(m_xBBatch * m_yBMJBatch[2] * m_tBatch); // [N,1]
    torch::Tensor A_I = -beamCharge * e6 / (denomConst * P1 * P2); // [N,M]
    torch::Tensor sqrI = bc(cI[0]) * cosn(0) + bc(cI[1]) * cosn(1)
            + bc(cI[2]) * cosn(2) + bc(cI[3]) * cosn(3)
            + bc(sI[1]) * sinn(1) + bc(sI[2]) * sinn(2); // [N,M]
    return bc(m_phaseSpaceBMJBatch) * A_I * sqrI; // [N,M]
}
