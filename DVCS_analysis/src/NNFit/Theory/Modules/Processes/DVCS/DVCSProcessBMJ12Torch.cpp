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

#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.h"

namespace {
const torch::TensorOptions kF64 = torch::TensorOptions().dtype(torch::kFloat64);
inline torch::Tensor zeroC() {
    return torch::complex(torch::zeros({}, kF64), torch::zeros({}, kF64));
}
} // namespace

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
// Kinematics setup (phi-independent; transcribed from DVCSProcessBMJ12)
// ---------------------------------------------------------------------------

void DVCSProcessBMJ12Torch::setupKinematicsTorch(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    // Populate the protected base kinematics m_xB, m_t, m_Q2, m_E, m_phi.
    setKinematics(kinematic);

    // Base-derived kinematics (DVCSProcessModule::initModule).
    const double M = PARTONS::Constant::PROTON_MASS;
    const double epsilon = 2 * m_xB * M / std::sqrt(m_Q2);
    const double y = m_Q2 / (2 * m_xB * M * m_E);
    const double eps2 = epsilon * epsilon;
    const double epsrootBase = std::sqrt(1 + eps2);
    const double tfactor = -m_Q2 / (4 * m_xB * (1 - m_xB) + eps2);
    m_tminBMJ = tfactor * (2 * (1 - m_xB) * (1 - epsrootBase) + eps2);
    m_tmaxBMJ = tfactor * (2 * (1 - m_xB) * (1 + epsrootBase) + eps2);

    // BMJ12 derived kinematics (DVCSProcessBMJ12::initModule).
    m_xB2 = m_xB * m_xB;
    m_Qpow[0] = std::sqrt(m_Q2);
    m_Qpow[1] = m_Q2;
    m_Qpow[2] = m_Qpow[0] * m_Q2;
    m_Qpow[3] = m_Q2 * m_Q2;
    m_Delta2[0] = m_t;
    m_Delta2[1] = m_t * m_t;
    m_xBtQ2[0] = 2 - m_xB + m_xB * m_t / m_Q2;
    m_xBtQ2[1] = m_xBtQ2[0] * m_xBtQ2[0];
    m_xBtQ2[2] = m_xBtQ2[1] * m_xBtQ2[0];
    m_M[0] = M;
    m_M[1] = M * M;
    m_yBMJ[0] = y;
    m_yBMJ[1] = y * y;
    m_yBMJ[2] = y * y * y;
    m_epsilonBMJ[0] = epsilon;
    m_epsilonBMJ[1] = eps2;
    m_epsroot[1] = 1 + eps2;
    m_epsroot[0] = std::sqrt(m_epsroot[1]);
    m_epsroot[2] = m_epsroot[1] * m_epsroot[0];
    m_epsroot[3] = m_epsroot[1] * m_epsroot[1];
    m_epsroot[4] = m_epsroot[2] * m_epsroot[1];
    m_epsroot[5] = m_epsroot[2] * m_epsroot[2];
    m_K[1] = -(m_t / m_Q2) * (1 - m_xB)
            * (1 - m_yBMJ[0] - m_yBMJ[1] * m_epsilonBMJ[1] / 4)
            * (1 - m_tminBMJ / m_t)
            * (m_epsroot[0]
                    + (4 * m_xB * (1 - m_xB) + m_epsilonBMJ[1])
                            / (4 * (1 - m_xB)) * (m_t - m_tminBMJ) / m_Q2);
    m_K[0] = std::sqrt(m_K[1]);
    m_Kt[1] = ((1 - m_xB) * m_xB + m_epsilonBMJ[1] / 4.) * (m_tminBMJ - m_t)
            * (m_t - m_tmaxBMJ) / m_Q2;
    m_Kt[0] = std::sqrt(m_Kt[1]);

    // Dirac and Pauli form factors (DVCSProcessBMJ12::computeFormFactors).
    m_F1 = (4. * m_M[1] - 2.79285 * m_t)
            / (std::pow(1. - 1.4084507042253522 * m_t, 2) * (4. * m_M[1] - 1. * m_t));
    m_F2 = (7.1714 * m_M[1])
            / (std::pow(1 - 1.4084507042253522 * m_t, 2) * (4 * m_M[1] - m_t));

    // CFF coefficients cF[j][0] (DVCSProcessBMJ12::computeCFFs).
    for (int j = 0; j < 2; j++) {
        m_cF[j] = (1. + (1 - 2 * j) * m_epsroot[0]) / (2. * m_epsroot[0])
                + (1 - m_xB) * m_xB2 * (4 * m_M[1] - m_t) * (1 + m_t / m_Q2)
                        / (m_Q2 * m_epsroot[0] * m_xBtQ2[1]);
    }
    {
        const double A = -std::sqrt(2.) * m_Kt[0]
                / (m_epsroot[0] * m_Qpow[0] * m_xBtQ2[0]);
        m_cF[2] = A * m_xB
                * (1 + 2. * m_xB * (4 * m_M[1] - m_t) / (m_Q2 * m_xBtQ2[0]));
    }

    // Phase space.
    m_phaseSpaceBMJ = m_xB * m_yBMJ[1]
            / (1024 * std::pow(PARTONS::Constant::PI, 5) * m_Qpow[3] * m_epsroot[0]);

    // Lepton-propagator pieces (phi-independent part).
    const double Delta2Q2 = m_t / m_Q2;
    m_yeps = m_yBMJ[0] * (1. + m_epsilonBMJ[1]);
    m_J = (1. - m_yBMJ[0] - m_yBMJ[0] * m_epsilonBMJ[1] / 2.) * (1. + Delta2Q2)
            - (1. - m_xB) * (2. - m_yBMJ[0]) * Delta2Q2;

    // ----- Unpolarized BH Fourier coefficients (computeFourierCoeffsBH) -----
    {
        const double F1 = m_F1, F2 = m_F2;
        const double F22 = std::pow(F2, 2);
        const double F1PlusF2 = F1 + F2;
        const double F1PlusF22 = std::pow(F1PlusF2, 2);
        const double Delta2M2 = m_t / (4 * m_M[1]);
        const double F1MinusDeltaF2 = std::pow(F1, 2) - Delta2M2 * F22;

        m_cBH0[0] = 8. * m_K[1]
                * ((2. + 3. * m_epsilonBMJ[1]) * F1MinusDeltaF2 / Delta2Q2
                        + 2. * m_xB2 * F1PlusF22)
                + std::pow((2. - m_yBMJ[0]), 2)
                        * ((2. + m_epsilonBMJ[1])
                                * ((m_xB2 / Delta2M2) * std::pow((1. + Delta2Q2), 2)
                                        + 4. * (1. - m_xB) * (1. + m_xB * Delta2Q2))
                                * F1MinusDeltaF2
                                + (4. * m_xB2)
                                        * (m_xB
                                                + (1. - m_xB + m_epsilonBMJ[1] / 2.)
                                                        * std::pow((1. - Delta2Q2), 2)
                                                - m_xB * (1. - 2. * m_xB)
                                                        * m_Delta2[1] / m_Qpow[3])
                                        * F1PlusF22)
                + 8. * (1. + m_epsilonBMJ[1])
                        * (1. - m_yBMJ[0] - m_epsilonBMJ[1] * m_yBMJ[1] / 4.)
                        * (2. * m_epsilonBMJ[1] * (1. - Delta2M2) * F1MinusDeltaF2
                                - m_xB2 * std::pow((1. - Delta2Q2), 2) * F1PlusF22);
        m_cBH0[1] = 8. * m_K[0] * (2. - m_yBMJ[0])
                * ((m_xB2 / Delta2M2 - 2. * m_xB - m_epsilonBMJ[1]) * F1MinusDeltaF2
                        + 2. * m_xB2 * (1. - (1. - 2. * m_xB) * Delta2Q2)
                                * F1PlusF22);
        m_cBH0[2] = 8. * m_xB2 * m_K[1]
                * (F1MinusDeltaF2 / Delta2M2 + 2. * F1PlusF22);
    }

    // ----- Interference angular coefficients (computeAngularCoeffsInterf) ----
    // Only the S = 0 blocks (m_C and m_S) are needed for the unpolarized target.
    {
        const double tQ2 = m_t / m_Q2;
        const double Kt2Q2 = m_Kt[1] / m_Q2;
        const double tp = m_t - m_tminBMJ;
        const double tpQ2 = tp / m_Q2;
        const double C2 = 1. - m_yBMJ[0] - m_epsilonBMJ[1] * m_yBMJ[1] / 4.;
        const double C1 = std::sqrt(C2);
        const double B2 = 2. - 2 * m_yBMJ[0] + m_yBMJ[1]
                + m_epsilonBMJ[1] * m_yBMJ[1] / 2.;

        double(*m_C)[3][4] = m_Cang; // local aliases matching the source names
        double(*m_S)[3][4] = m_Sang;

        // C++ coefficients
        m_C[0][0][0] = -4 * (2. - m_yBMJ[0]) * (1. + m_epsroot[0]) / m_epsroot[3]
                * (Kt2Q2 * (2. - m_yBMJ[0]) * (2. - m_yBMJ[0]) / m_epsroot[0]
                        + tQ2 * C2 * (2 - m_xB)
                                * (1.
                                        + (2 * m_xB
                                                * (2 - m_xB + (m_epsroot[0] - 1.) / 2.
                                                        + m_epsilonBMJ[1] / (2 * m_xB))
                                                * tQ2 + m_epsilonBMJ[1])
                                                / ((2. - m_xB) * (1 + m_epsroot[0]))));
        m_C[0][1][0] = 8 * (2. - m_yBMJ[0]) * m_xB * tQ2 / m_epsroot[3]
                * (Kt2Q2 * (2. - m_yBMJ[0]) * (2. - m_yBMJ[0]) / m_epsroot[0]
                        + C2 * (1 + m_epsroot[0]) / 2. * (1. + tQ2)
                                * (1.
                                        + (m_epsroot[0] - 1. + 2 * m_xB) * tQ2
                                                / (1 + m_epsroot[0])));
        m_C[0][2][0] = 8 * (2. - m_yBMJ[0]) * tQ2 / m_epsroot[3]
                * (Kt2Q2 * (2. - m_yBMJ[0]) * (2. - m_yBMJ[0]) / m_epsroot[0]
                        * (1 + m_epsroot[0] - 2 * m_xB) / 2.
                        - C2
                                * (2 * Kt2Q2
                                        - (1 + m_epsroot[0]) / 2.
                                                * (1 + m_epsroot[0] - m_xB
                                                        + (m_epsroot[0] - 1.
                                                                + m_xB * (3. + m_epsroot[0]
                                                                        - 2 * m_xB)
                                                                        / (1 + m_epsroot[0]))
                                                                * tQ2)));
        m_C[0][0][1] = -16 * m_K[0] * C2 / m_epsroot[4]
                * ((1. + (1. - m_xB) * (m_epsroot[0] - 1.) / (2 * m_xB)
                        + m_epsilonBMJ[1] / (4 * m_xB)) * m_xB * tQ2
                        - 3 * m_epsilonBMJ[1] / 4.)
                - 4 * m_K[0] * B2 * (1. + m_epsroot[0] - m_epsilonBMJ[1]) / m_epsroot[4]
                        * (1. - (1. - 3 * m_xB) * tQ2
                                + (1. - m_epsroot[0] + 3 * m_epsilonBMJ[1]) * m_xB * tQ2
                                        / (1. + m_epsroot[0] - m_epsilonBMJ[1]));
        m_C[0][1][1] = 16 * m_K[0] * m_xB * tQ2 / m_epsroot[4]
                * ((2. - m_yBMJ[0]) * (2. - m_yBMJ[0]) * (1. - (1. - 2 * m_xB) * tQ2)
                        + C2 * (1 + m_epsroot[0] - 2 * m_xB) * tpQ2 / 2.);
        m_C[0][2][1] = -16 * m_K[0] * tQ2 / m_epsroot[3]
                * (C2
                        * (1. - (1. - 2 * m_xB) * tQ2
                                + (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1]) * tpQ2
                                        / (4 * m_epsroot[0]))
                        - (2. - m_yBMJ[0]) * (2. - m_yBMJ[0])
                                * (1. - m_xB / 2.
                                        + (1 + m_epsroot[0] - 2 * m_xB) / 4. * (1. - tQ2)
                                        + (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1])
                                                * tpQ2 / (2 * m_epsroot[0])));
        m_C[0][0][2] = 8 * (2. - m_yBMJ[0]) * C2 / m_epsroot[3]
                * (2 * m_epsilonBMJ[1] * Kt2Q2 / (m_epsroot[0] + m_epsroot[1])
                        + m_xB * tQ2 * tpQ2
                                * (1. - m_xB - (m_epsroot[0] - 1.) / 2.
                                        + m_epsilonBMJ[1] / (2 * m_xB)));
        m_C[0][1][2] = 8 * (2. - m_yBMJ[0]) * C2 * m_xB * tQ2 / m_epsroot[3]
                * (4 * Kt2Q2 / m_epsroot[0]
                        + (1. + m_epsroot[0] - 2 * m_xB) / 2. * (1 + tQ2) * tpQ2);
        m_C[0][2][2] = 4 * (2. - m_yBMJ[0]) * C2 * tQ2 / m_epsroot[3]
                * (4 * (1. - 2 * m_xB) * Kt2Q2 / m_epsroot[0]
                        - (3. - m_epsroot[0] - 2 * m_xB + m_epsilonBMJ[1] / m_xB)
                                * m_xB * tpQ2);
        m_C[0][0][3] = -8 * m_K[0] * C2 * (m_epsroot[0] - 1.) / m_epsroot[4]
                * ((1. - m_xB) * tQ2 + (m_epsroot[0] - 1.) / 2. * (1 + tQ2));
        m_C[0][1][3] = -8 * m_K[0] * C2 * m_xB * tQ2 / m_epsroot[4]
                * (m_epsroot[0] - 1. + (1. + m_epsroot[0] - 2 * m_xB) * tQ2);
        m_C[0][2][3] = 16 * m_K[0] * C2 * tQ2 * tpQ2 / m_epsroot[4]
                * (m_xB * (1. - m_xB) + m_epsilonBMJ[1] / 4.);

        // S++ coefficients
        m_S[0][0][1] = 8 * m_K[0] * (2. - m_yBMJ[0]) * m_yBMJ[0] / m_epsroot[1]
                * (1. + (1. - m_xB + (m_epsroot[0] - 1.) / 2.) / m_epsroot[1] * tpQ2);
        m_S[0][1][1] = -8 * m_K[0] * (2. - m_yBMJ[0]) * m_yBMJ[0] * m_xB * tQ2
                / m_epsroot[3]
                * (m_epsroot[0] - 1. + (1. + m_epsroot[0] - 2 * m_xB) * tQ2);
        m_S[0][2][1] = 8 * m_K[0] * (2. - m_yBMJ[0]) * m_yBMJ[0] * tQ2 / m_epsroot[1]
                * (1.
                        - (1. - 2 * m_xB) * (1. + m_epsroot[0] - 2 * m_xB)
                                / (2 * m_epsroot[1]) * tpQ2);
        m_S[0][0][2] = -4 * C2 * m_yBMJ[0] / m_epsroot[2]
                * (1. + m_epsroot[0] - 2 * m_xB) * tpQ2
                * ((m_epsilonBMJ[1] - m_xB * (m_epsroot[0] - 1.))
                        / (1. + m_epsroot[0] - 2 * m_xB)
                        - (2 * m_xB + m_epsilonBMJ[1]) * tpQ2 / (2 * m_epsroot[0]));
        m_S[0][1][2] = -4 * C2 * m_yBMJ[0] * m_xB * tQ2 / m_epsroot[3]
                * (1. - (1. - 2 * m_xB) * tQ2)
                * (m_epsroot[0] - 1. + (1. + m_epsroot[0] - 2 * m_xB) * tQ2);
        m_S[0][2][2] = -8 * C2 * m_yBMJ[0] * tQ2 * tpQ2 / m_epsroot[3]
                * (1. - m_xB / 2. + 3 * m_epsilonBMJ[1] / 4.)
                * (1. + m_epsroot[0] - 2 * m_xB)
                * (1.
                        + (4 * (1 - m_xB) * m_xB + m_epsilonBMJ[1])
                                / (4. - 2 * m_xB + 3 * m_epsilonBMJ[1]) * tQ2);

        // C0+
        m_C[2][0][0] = 12 * std::sqrt(2.) * m_K[0] * (2. - m_yBMJ[0]) * C1 / m_epsroot[4]
                * (m_epsilonBMJ[1] + (2. - 6 * m_xB - m_epsilonBMJ[1]) / 3. * tQ2);
        m_C[2][1][0] = 24 * std::sqrt(2.) * m_K[0] * (2. - m_yBMJ[0]) * C1 * m_xB * tQ2
                / m_epsroot[4] * (1. - (1. - 2 * m_xB) * tQ2);
        m_C[2][2][0] = 4 * std::sqrt(2.) * m_K[0] * (2. - m_yBMJ[0]) * C1 * tQ2
                / m_epsroot[4] * (8. - 6 * m_xB + 5 * m_epsilonBMJ[1])
                * (1.
                        - tQ2 * (2. - 12 * m_xB * (1. - m_xB) - m_epsilonBMJ[1])
                                / (8. - 6 * m_xB + 5 * m_epsilonBMJ[1]));
        m_C[2][0][1] = 8 * std::sqrt(2.) * C1 / m_epsroot[3]
                * ((2. - m_yBMJ[0]) * (2. - m_yBMJ[0]) * tpQ2
                        * (1. - m_xB
                                + ((1. - m_xB) * m_xB + m_epsilonBMJ[1] / 4.)
                                        / m_epsroot[0] * tpQ2)
                        + C2 / m_epsroot[0] * (1. - (1. - 2 * m_xB) * tQ2)
                                * (m_epsilonBMJ[1]
                                        - 2 * (1. + m_epsilonBMJ[1] / (2 * m_xB))
                                                * m_xB * tQ2));
        m_C[2][1][1] = 16 * std::sqrt(2.) * C1 * m_xB * tQ2 / m_epsroot[4]
                * (Kt2Q2 * (2. - m_yBMJ[0]) * (2. - m_yBMJ[0])
                        + std::pow(1. - (1. - 2 * m_xB) * tQ2, 2) * C2);
        m_C[2][2][1] = 8 * std::sqrt(2.) * C1 * tQ2 / m_epsroot[4]
                * (Kt2Q2 * (1. - 2 * m_xB) * (2. - m_yBMJ[0]) * (2. - m_yBMJ[0])
                        + (1. - (1. - 2 * m_xB) * tQ2) * C2
                                * (4. - 2 * m_xB + 3 * m_epsilonBMJ[1]
                                        + tQ2 * (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1])));
        m_C[2][0][2] = -8 * std::sqrt(2.) * m_K[0] * (2. - m_yBMJ[0]) * C1 / m_epsroot[4]
                * (1. + m_epsilonBMJ[1] / 2.)
                * (1.
                        + (1. + m_epsilonBMJ[1] / (2 * m_xB))
                                / (1. + m_epsilonBMJ[1] / 2.) * m_xB * tQ2);
        m_C[2][1][2] = 8 * std::sqrt(2.) * m_K[0] * (2. - m_yBMJ[0]) * C1 * m_xB * tQ2
                / m_epsroot[4] * (1. - (1. - 2 * m_xB) * tQ2);
        m_C[2][2][2] = 8 * std::sqrt(2.) * m_K[0] * (2. - m_yBMJ[0]) * C1 * tQ2
                / m_epsroot[3]
                * (1. - m_xB
                        + tpQ2 / 2. * (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1])
                                / m_epsroot[0]);

        // S0+
        m_S[2][0][1] = 8 * std::sqrt(2.) * (2. - m_yBMJ[0]) * m_yBMJ[0] * C1 * Kt2Q2
                / m_epsroot[3];
        m_S[2][1][1] = 4 * std::sqrt(2.) * (2. - m_yBMJ[0]) * m_yBMJ[0] * C1 * m_xB * tQ2
                / m_epsroot[3]
                * (4 * (1. - m_xB) * tQ2 * (1 + m_xB * tQ2)
                        + m_epsilonBMJ[1] * (1 + tQ2) * (1 + tQ2));
        m_S[2][2][1] = -8 * std::sqrt(2.) * (2. - m_yBMJ[0]) * m_yBMJ[0]
                * (1. - 2 * m_xB) * C1 * tQ2 * Kt2Q2 / m_epsroot[3];
        m_S[2][0][2] = 8 * std::sqrt(2.) * m_K[0] * m_yBMJ[0] * C1
                * (1. + m_epsilonBMJ[1] / 2.) / m_epsroot[3]
                * (1.
                        + (1. + m_epsilonBMJ[1] / (2 * m_xB))
                                / (1. + m_epsilonBMJ[1] / 2.) * m_xB * tQ2);
        m_S[2][1][2] = -8 * std::sqrt(2.) * m_K[0] * m_yBMJ[0] * C1 * m_xB * tQ2
                / m_epsroot[3] * (1. - (1. - 2 * m_xB) * tQ2);
        m_S[2][2][2] = -2 * std::sqrt(2.) * m_K[0] * m_yBMJ[0] * C1 * tQ2 / m_epsroot[3]
                * (4. - 4 * m_xB + 2 * m_epsilonBMJ[1]
                        + 2 * tQ2 * (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1]));

        // C-+
        m_C[1][0][0] = 8 * (2. - m_yBMJ[0]) / m_epsroot[2]
                * ((2. - m_yBMJ[0]) * (2. - m_yBMJ[0]) * (m_epsroot[0] - 1.)
                        / (2 * m_epsroot[1]) * Kt2Q2
                        + C2 / m_epsroot[0]
                                * (1. - m_xB - (m_epsroot[0] - 1.) / 2.
                                        + m_epsilonBMJ[1] / (2 * m_xB)) * m_xB * tQ2
                                * tpQ2);
        m_C[1][1][0] = 4 * (2. - m_yBMJ[0]) * m_xB * tQ2 / m_epsroot[4]
                * (2 * Kt2Q2 * B2
                        - (1. - (1. - 2 * m_xB) * tQ2) * C2
                                * (m_epsroot[0] - 1.
                                        + (m_epsroot[0] + 1. - 2 * m_xB) * tQ2));
        m_C[1][2][0] = 4 * (2. - m_yBMJ[0]) * tQ2 / m_epsroot[3]
                * (tpQ2 * C2
                        * (2 * m_xB2 - m_epsilonBMJ[1] - 3 * m_xB + m_xB * m_epsroot[0])
                        + Kt2Q2 / m_epsroot[0]
                                * (4.
                                        - 2 * m_xB * (2. - m_yBMJ[0]) * (2. - m_yBMJ[0])
                                                - 4 * m_yBMJ[0]
                                        + m_yBMJ[1] - m_yBMJ[1] * m_epsroot[2]));
        m_C[1][0][1] = 8 * m_K[0] / m_epsroot[2]
                * ((2. - m_yBMJ[0]) * (2. - m_yBMJ[0]) * (2. - m_epsroot[0]) / m_epsroot[1]
                        * ((m_epsroot[0] - 1. + m_epsilonBMJ[1]) / (2 * (2. - m_epsroot[0]))
                                * (1. - tQ2) - m_xB * tQ2)
                        + 2 * C2 / m_epsroot[0]
                                * ((1. - m_epsroot[0] + m_epsilonBMJ[1] / 2.)
                                        / (2 * m_epsroot[0])
                                        + tQ2
                                                * (1. - 3 * m_xB / 2.
                                                        + (m_xB + m_epsilonBMJ[1] / 2.)
                                                                / (2 * m_epsroot[0]))));
        m_C[1][1][1] = 8 * m_K[0] * m_xB * tQ2 / m_epsroot[4]
                * (2 * (1. - (1. - 2 * m_xB) * tQ2) * B2
                        + C2
                                * (3. - m_epsroot[0]
                                        - (3 * (1. - 2 * m_xB) + m_epsroot[0]) * tQ2));
        m_C[1][2][1] = 4 * m_K[0] * tQ2 / m_epsroot[4]
                * (B2
                        * (5. - 4 * m_xB + 3 * m_epsilonBMJ[1] - m_epsroot[0]
                                - tQ2
                                        * (1. - m_epsilonBMJ[1] - m_epsroot[0]
                                                - 2 * m_xB
                                                        * (4. - 4 * m_xB - m_epsroot[0])))
                        + C2
                                * (8. + 5 * m_epsilonBMJ[1] - 6 * m_xB
                                        + 2 * m_xB * m_epsroot[0]
                                        - tQ2
                                                * (2. - m_epsilonBMJ[1] + 2 * m_epsroot[0]
                                                        - 4 * m_xB
                                                                * (3. - 3 * m_xB
                                                                        + m_epsroot[0]))));
        m_C[1][0][2] = 4 * (2. - m_yBMJ[0]) * C2 * (1. + m_epsroot[0]) / m_epsroot[4]
                * ((2. - 3 * m_xB) * tQ2
                        + (1. - 2 * m_xB + 2 * (1. - m_xB) / (1. + m_epsroot[0]))
                                * m_xB * tQ2 * tQ2
                        + (1.
                                + (m_epsroot[0] + m_xB + (1. - m_xB) * tQ2)
                                        / (1. + m_epsroot[0]) * tQ2)
                                * m_epsilonBMJ[1]);
        m_C[1][1][2] = 4 * (2. - m_yBMJ[0]) * C2 * m_xB * tQ2 / m_epsroot[4]
                * (4 * Kt2Q2 + 1. + m_epsroot[0]
                        + tQ2
                                * ((1. - 2 * m_xB) * (1. - 2 * m_xB - m_epsroot[0]) * tQ2
                                        - 2. + 4 * m_xB + 2 * m_xB * m_epsroot[0]));
        m_C[1][2][2] = 16 * (2. - m_yBMJ[0]) * C2 * tQ2 / m_epsroot[2]
                * (Kt2Q2 * (1. - 2 * m_xB) / m_epsroot[1]
                        - (1. - m_xB) / (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1])
                                * (2 * m_xB2 - m_epsilonBMJ[1] - 3 * m_xB
                                        - m_xB * m_epsroot[0])
                        - tpQ2
                                * (2 * m_xB2 - m_epsilonBMJ[1] - 3 * m_xB
                                        - m_xB * m_epsroot[0]) / (4 * m_epsroot[0]));
        m_C[1][0][3] = -8 * m_K[0] * C2 * (1. + m_epsroot[0] + m_epsilonBMJ[1] / 2.)
                / m_epsroot[4]
                * (1.
                        + (1. + m_epsroot[0] + m_epsilonBMJ[1] / (2 * m_xB))
                                / (1. + m_epsroot[0] + m_epsilonBMJ[1] / 2) * m_xB * tQ2);
        m_C[1][1][3] = 8 * m_K[0] * C2 * m_xB * tQ2 / m_epsroot[4] * (1. + m_epsroot[0])
                * (1. - tQ2 * (1. - 2 * m_xB - m_epsroot[0]) / (1. + m_epsroot[0]));
        m_C[1][2][3] = 16 * m_K[0] * C2 * tQ2 / m_epsroot[3]
                * (1. - m_xB
                        + tpQ2 * (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1])
                                / (4 * m_epsroot[0]));

        // S-+
        m_S[1][0][1] = 4 * m_K[0] * (2. - m_yBMJ[0]) * m_yBMJ[0] / m_epsroot[3]
                * (1. - m_epsroot[0] + 2 * m_epsilonBMJ[1]
                        - 2 * (1. + (m_epsroot[0] - 1.) / (2 * m_xB)) * m_xB * tQ2);
        m_S[1][1][1] = 8 * m_K[0] * (2. - m_yBMJ[0]) * m_yBMJ[0] * m_xB * tQ2
                / m_epsroot[3] * (1. + m_epsroot[0])
                * (1. - tQ2 * (1. - 2 * m_xB - m_epsroot[0]) / (1. + m_epsroot[0]));
        m_S[1][2][1] = 4 * m_K[0] * (2. - m_yBMJ[0]) * m_yBMJ[0] * tQ2 / m_epsroot[3]
                * (3. + 2 * m_epsilonBMJ[1] + m_epsroot[0]
                        - 2 * m_xB * (1. + m_epsroot[0])
                        - tQ2 * (1. - 2 * m_xB) * (1. - 2 * m_xB - m_epsroot[0]));
        m_S[1][0][2] = 2 * m_yBMJ[0] * C2 * (1. + m_epsroot[0]) / m_epsroot[3]
                * (m_epsilonBMJ[1] - 2 * (1. + m_epsilonBMJ[1] / (2 * m_xB)) * m_xB * tQ2)
                * (1. + (m_epsroot[0] - 1. + 2 * m_xB) / (1. + m_epsroot[0]) * tQ2);
        m_S[1][1][2] = 4 * m_yBMJ[0] * C2 * m_xB * tQ2 / m_epsroot[3]
                * (1. + m_epsroot[0]) * (1. - (1. - 2 * m_xB) * tQ2)
                * (1. - tQ2 * (1. - 2 * m_xB - m_epsroot[0]) / (1. + m_epsroot[0]));
        m_S[1][2][2] = 2 * m_yBMJ[0] * C2 * tQ2 / m_epsroot[3]
                * (4. - 2 * m_xB + 3 * m_epsilonBMJ[1]
                        + tQ2 * (4 * m_xB * (1. - m_xB) + m_epsilonBMJ[1]))
                * (1. + m_epsroot[0] - tQ2 * (1. - 2 * m_xB - m_epsroot[0]));
    }

    // ----- CFFs from the neural network (tensors) --------------------------
    DVCSCFFNNTorch* pCFF =
            dynamic_cast<DVCSCFFNNTorch*>(m_pConvolCoeffFunctionModule);
    if (!pCFF) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSCFFNNTorch convol-coeff module.");
    }
    const double xi = m_xB / (2. - m_xB);
    pCFF->setupKinematicsTorch(xi, m_t, m_Q2);
    DVCSCFFNNTorch::AllCFFsTensor cffs = pCFF->computeAllCFFsTensor();
    m_CFFstd[0] = cffs.H;
    m_CFFstd[1] = cffs.E;
    m_CFFstd[2] = cffs.Ht;
    m_CFFstd[3] = cffs.Et;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            m_CFF[i][j] = m_cF[j] * m_CFFstd[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor CFF layer
// ---------------------------------------------------------------------------

torch::Tensor DVCSProcessBMJ12Torch::cffTensor(int F, int a, int b) const {
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
        return zeroC();
    }
    return m_CFF[F][j];
}

torch::Tensor DVCSProcessBMJ12Torch::C_VCS0(int a1, int b1, int a2, int b2) const {
    torch::Tensor H1 = cffTensor(0, a1, b1), E1 = cffTensor(1, a1, b1),
            Ht1 = cffTensor(2, a1, b1), Et1 = cffTensor(3, a1, b1);
    torch::Tensor H2 = cffTensor(0, a2, b2), E2 = cffTensor(1, a2, b2),
            Ht2 = cffTensor(2, a2, b2), Et2 = cffTensor(3, a2, b2);

    const double tQ2 = m_t / m_Q2;
    return 4. * (1 - m_xB) * (1 + m_xB * tQ2) / m_xBtQ2[1]
            * (H1 * torch::conj(H2) + Ht1 * torch::conj(Ht2))
            + (2 + tQ2) * m_epsilonBMJ[1] / m_xBtQ2[1] * Ht1 * torch::conj(Ht2)
            - m_t / (4 * m_M[1]) * E1 * torch::conj(E2)
            - m_xB2 / m_xBtQ2[1]
                    * (std::pow(1 + tQ2, 2)
                            * (H1 * torch::conj(E2) + E1 * torch::conj(H2)
                                    + E1 * torch::conj(E2))
                            + Ht1 * torch::conj(Et2) + Et1 * torch::conj(Ht2)
                            + m_t / (4 * m_M[1]) * Et1 * torch::conj(Et2));
}

torch::Tensor DVCSProcessBMJ12Torch::C_VCS0(int a1, int b1, int a2, int b2,
        int a3, int b3) const {
    // S = 0 -> signe = +1
    return C_VCS0(a1, b1, a2, b2) + C_VCS0(a1, b1, a3, b3);
}

torch::Tensor DVCSProcessBMJ12Torch::C_VCS0(int a1, int b1, int a2, int b2,
        int a3, int b3, int a4, int b4) const {
    // S = 0 -> signe = +1
    return C_VCS0(a1, b1, a2, b2) + C_VCS0(a3, b3, a4, b4);
}

torch::Tensor DVCSProcessBMJ12Torch::C_I0(int a, int b,
        const std::string& VA) const {
    torch::Tensor H = cffTensor(0, a, b), E = cffTensor(1, a, b),
            Ht = cffTensor(2, a, b), Et = cffTensor(3, a, b);

    const double F1PlusF2 = m_F1 + m_F2;
    const double xBF1PlusF2 = m_xB * F1PlusF2 / m_xBtQ2[0];

    if (VA == "V") {
        return xBF1PlusF2 * (H + E);
    } else if (VA == "A") {
        return xBF1PlusF2 * Ht;
    }
    return m_F1 * H - m_t / (4 * m_M[1]) * m_F2 * E + xBF1PlusF2 * Ht;
}

torch::Tensor DVCSProcessBMJ12Torch::C_I0n(unsigned int n, int a, int b) const {
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
        return zeroC();
    }
    if (n >= 4) {
        return zeroC();
    }
    return m_Cang[i][0][n] * C_I0(a, b, "") + m_Cang[i][1][n] * C_I0(a, b, "V")
            + m_Cang[i][2][n] * C_I0(a, b, "A");
}

torch::Tensor DVCSProcessBMJ12Torch::S_I0n(unsigned int n, int a, int b) const {
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
        return zeroC();
    }
    if (n >= 4) {
        return zeroC();
    }
    return m_Sang[i][0][n] * C_I0(a, b, "") + m_Sang[i][1][n] * C_I0(a, b, "V")
            + m_Sang[i][2][n] * C_I0(a, b, "A");
}

// ---------------------------------------------------------------------------
// Cross section sigma(lambda, phi), batched over phi
// ---------------------------------------------------------------------------

// Sub-process atoms — tensor siblings of CrossSectionBH / CrossSectionVCS /
// CrossSectionInterf. Each reads the phi-independent state cached by
// setupKinematicsTorch() (run once by the base crossSectionTensor() template
// method) and assembles its phi-dependent contribution, exactly as the scalar
// component methods read the members set by setKinematics().

torch::Tensor DVCSProcessBMJ12Torch::crossSectionBHTensor(double beamHelicity,
        double beamCharge, const torch::Tensor& phi) {

    // BH is independent of beam helicity and charge for the unpolarized target.
    (void) beamHelicity;
    (void) beamCharge;

    const double e6 = std::pow(PARTONS::Constant::POSITRON_CHARGE, 6);

    torch::Tensor phi1 = PARTONS::Constant::PI - phi; // BMK angle, Trento convention
    auto cosn = [&](int n) { return torch::cos(double(n) * phi1); };

    torch::Tensor P1 = -(m_J + 2 * m_K[0] * torch::cos(phi1)) / m_yeps;
    torch::Tensor P2 = (1. + m_t / m_Q2) - P1;

    torch::Tensor sqrBH = m_cBH0[0] * cosn(0) + m_cBH0[1] * cosn(1)
            + m_cBH0[2] * cosn(2);
    torch::Tensor A_BH = e6
            / (m_xB2 * m_yBMJ[1] * m_epsroot[3] * m_t * P1 * P2);
    return m_phaseSpaceBMJ * A_BH * sqrBH;
}

torch::Tensor DVCSProcessBMJ12Torch::crossSectionVCSTensor(double beamHelicity,
        double beamCharge, const torch::Tensor& phi) {

    (void) beamCharge; // VCS is independent of beam charge.

    const double lambda = beamHelicity;
    const double e6 = std::pow(PARTONS::Constant::POSITRON_CHARGE, 6);

    // ----- VCS unpolarized Fourier coefficients (0-d real tensors) ---------
    const double C2 = 1. - m_yBMJ[0] - m_epsilonBMJ[1] * m_yBMJ[1] / 4.;
    const double C1 = std::sqrt(C2);
    torch::Tensor cVCS0 = (2 * (2 - 2 * m_yBMJ[0]
            + m_yBMJ[1] * (1. + m_epsilonBMJ[1] / 2.))
            * torch::real(C_VCS0(1, 1, 1, 1, -1, 1, -1, 1))
            + 8 * C2 * torch::real(C_VCS0(0, 1, 0, 1))) / m_epsroot[1];
    torch::Tensor cVCS1 = 4 * std::sqrt(2.) * C1 / m_epsroot[1] * (2 - m_yBMJ[0])
            * torch::real(C_VCS0(0, 1, 1, 1, -1, 1));
    torch::Tensor sVCS1 = 4 * std::sqrt(2.) * C1 / m_epsroot[1]
            * (-lambda * m_yBMJ[0] * m_epsroot[0])
            * torch::imag(C_VCS0(0, 1, 1, 1, -1, 1));
    torch::Tensor cVCS2 = 8 * C2 / m_epsroot[1] * torch::real(C_VCS0(-1, 1, 1, 1));

    torch::Tensor phi1 = PARTONS::Constant::PI - phi;
    auto cosn = [&](int n) { return torch::cos(double(n) * phi1); };
    auto sinn = [&](int n) { return torch::sin(double(n) * phi1); };

    const double A_VCS = e6 / (m_yBMJ[1] * m_Q2);
    torch::Tensor sqrVCS = cVCS0 * cosn(0) + cVCS1 * cosn(1) + cVCS2 * cosn(2)
            + sVCS1 * sinn(1);
    return m_phaseSpaceBMJ * A_VCS * sqrVCS;
}

torch::Tensor DVCSProcessBMJ12Torch::crossSectionInterfTensor(
        double beamHelicity, double beamCharge, const torch::Tensor& phi) {

    const double lambda = beamHelicity;
    const double e6 = std::pow(PARTONS::Constant::POSITRON_CHARGE, 6);

    // ----- Interference unpolarized Fourier coefficients -------------------
    torch::Tensor cI[4];
    for (unsigned int n = 0; n < 4; n++) {
        cI[n] = torch::real(C_I0n(n, 1, 1) + C_I0n(n, -1, 1) + C_I0n(n, 0, 1));
    }
    torch::Tensor sI[4] = { zeroC(), zeroC(), zeroC(), zeroC() };
    for (unsigned int n = 1; n < 3; n++) {
        sI[n] = lambda
                * torch::imag(S_I0n(n, 1, 1) + S_I0n(n, -1, 1) + S_I0n(n, 0, 1));
    }

    torch::Tensor phi1 = PARTONS::Constant::PI - phi;
    auto cosn = [&](int n) { return torch::cos(double(n) * phi1); };
    auto sinn = [&](int n) { return torch::sin(double(n) * phi1); };

    torch::Tensor P1 = -(m_J + 2 * m_K[0] * torch::cos(phi1)) / m_yeps;
    torch::Tensor P2 = (1. + m_t / m_Q2) - P1;

    torch::Tensor A_I = -beamCharge * e6 / (m_xB * m_yBMJ[2] * m_t * P1 * P2);
    torch::Tensor sqrI = cI[0] * cosn(0) + cI[1] * cosn(1) + cI[2] * cosn(2)
            + cI[3] * cosn(3) + sI[1] * sinn(1) + sI[2] * sinn(2);
    return m_phaseSpaceBMJ * A_I * sqrI;
}
