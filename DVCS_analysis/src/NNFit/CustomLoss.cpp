/**
 * @file CustomLoss.cpp
 *
 * Implements the χ² loss over DVCS observable data. Per-batch forward()
 * loops over rows, building one DVCSObservableKinematic per data point
 * and driving the PARTONS-registered tensor module chain to obtain a
 * differentiable A_LU^{sin1φ} prediction. Per-point contributions
 *
 *     ((pred − y_obs) / sigma)²
 *
 * are accumulated into a single 0-d tensor whose gradient flows back
 * through the entire physics chain to the NN parameters.
 */

#include "NNFit/CustomLoss.h"

#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <partons/modules/observable/DVCS/DVCSObservable.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/modules/scales/DVCS/DVCSScalesModule.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterModule.h>
#include <partons/modules/scales/DVCS/DVCSScalesQ2Multiplier.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterXBToXi.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>

#include "NNFit/theory/Modules/CFFs/DVCS/DVCSCFFNNPytorch.h"
#include "NNFit/theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.h"
#include "NNFit/theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

CustomLossImpl::CustomLossImpl(
        CFFNNModel net,
        const std::vector<std::string>& output_layer)
        : m_pCFF(nullptr), m_pProcess(nullptr), m_pObs(nullptr) {

    using namespace PARTONS;

    // ─── PARTONS tensor module chain (factory-instantiated) ────────────────
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()
                ->newDVCSConvolCoeffFunctionModule(DVCSCFFNNPytorch::classId);
    m_pCFF = static_cast<DVCSCFFNNPytorch*>(pDVCSCFF);
    m_pCFF->setModel(net, output_layer);
    m_pCFF->setQCDOrderType(PerturbativeQCDOrderType::LO);

    DVCSXiConverterModule* pDVCSXiConverter =
            Partons::getInstance()->getModuleObjectFactory()
                ->newDVCSXiConverterModule(DVCSXiConverterXBToXi::classId);

    DVCSScalesModule* pDVCSScales =
            Partons::getInstance()->getModuleObjectFactory()
                ->newDVCSScalesModule(DVCSScalesQ2Multiplier::classId);

    DVCSProcessModule* pDVCSProcess =
            Partons::getInstance()->getModuleObjectFactory()
                ->newDVCSProcessModule(DVCSProcessBMJ12Torch::classId);
    m_pProcess = static_cast<DVCSProcessBMJ12Torch*>(pDVCSProcess);
    m_pProcess->setXiConverterModule(pDVCSXiConverter);
    m_pProcess->setScaleModule(pDVCSScales);
    m_pProcess->setConvolCoeffFunctionModule(pDVCSCFF);

    DVCSObservable* pDVCSObs =
            Partons::getInstance()->getModuleObjectFactory()
                ->newDVCSObservable(DVCSAluMinusSin1PhiTorch::classId);
    m_pObs = static_cast<DVCSAluMinusSin1PhiTorch*>(pDVCSObs);
    m_pObs->setProcessModule(pDVCSProcess);
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

torch::Tensor CustomLossImpl::forward(
        const torch::Tensor& X,
        const torch::Tensor& E,
        const torch::Tensor& y_obs,
        const torch::Tensor& sigma) {

    const int64_t N = X.size(0);

    // Accumulator initialised as a 0-d zero tensor in the same dtype/device
    // as y_obs so the result type is consistent with the data.
    torch::Tensor chi2 = torch::zeros({}, y_obs.options());

    for (int64_t i = 0; i < N; ++i) {

        // Build kinematic for this data point. φ slot is irrelevant —
        // the observable integrates it out, but the constructor requires
        // a value, so we just pass 0.
        const double xB = X[i][0].item<double>();
        const double t  = X[i][1].item<double>();
        const double Q2 = X[i][2].item<double>();
        const double Ei = E[i].item<double>();

        PARTONS::DVCSObservableKinematic kin(xB, t, Q2, Ei, 0.);

        // Differentiable A_LU^{sin1φ} prediction (0-d tensor with autograd).
        torch::Tensor pred = m_pObs->computeTensor(kin);

        // χ² contribution: ((pred − y_obs[i]) / sigma[i])²
        torch::Tensor resid = (pred - y_obs[i]) / sigma[i];
        chi2 = chi2 + resid * resid;
    }

    return chi2;
}