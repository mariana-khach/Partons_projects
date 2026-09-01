//
// Created by Mariana Khachatryan on 6/17/26.
//

#include "../../include/NNFit/CustomLoss.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <partons/modules/observable/DVCS/DVCSObservable.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/modules/scales/DVCS/DVCSScalesQ2Multiplier.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterXBToXi.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/ServiceObjectRegistry.h>

#include "../../include/NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.h"
#include "../../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.h"
#include "../../include/NNFit/Theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.h"
#include "../../include/NNFit/Theory/Modules/Services/DVCS/DVCSObservableServiceTorch.h"

namespace {
const torch::TensorOptions kF64 = torch::TensorOptions().dtype(torch::kFloat64);
} // namespace

CustomLossImpl::CustomLossImpl(CFFNNModel net,
        const std::vector<std::string>& outputLayer, const torch::Tensor& xMin,
        const torch::Tensor& xMax, double xPow, bool normalize)
        : m_normalize(normalize) {

    using namespace PARTONS;

    // Build the *Torch module chain once — same wiring as observ_calc_torch().
    DVCSConvolCoeffFunctionModule* pCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNTorch::classId);
    static_cast<DVCSCFFNNTorch*>(pCFF)->setModel(net, outputLayer, xMin, xMax, xPow);

    DVCSXiConverterModule* pXi =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSXiConverterModule(
                    DVCSXiConverterXBToXi::classId);

    DVCSScalesModule* pScales =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSScalesModule(
                    DVCSScalesQ2Multiplier::classId);

    DVCSProcessModule* pProc =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSProcessModule(
                    DVCSProcessBMJ12Torch::classId);

    DVCSObservable* pObs =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    DVCSAluMinusSin1PhiTorch::classId);

    pCFF->setQCDOrderType(PerturbativeQCDOrderType::LO);

    pProc->setXiConverterModule(pXi);
    pProc->setScaleModule(pScales);
    pProc->setConvolCoeffFunctionModule(pCFF);
    pObs->setProcessModule(pProc);

    // Torch-aware service, fetched by name through the standard registry.
    m_pServiceTorch = static_cast<DVCSObservableServiceTorch*>(
            Partons::getInstance()->getServiceObjectRegistry()->get(
                    "DVCSObservableServiceTorch"));

    // Base tensor-observable handle (cross-cast: separate base subobject).
    m_pObsTorch = dynamic_cast<DVCSObservableTorch*>(pObs);
    if (!m_pObsTorch) {
        throw ElemUtils::CustomException("CustomLossImpl", __func__,
                "Wired observable is not a DVCSObservableTorch.");
    }
}

torch::Tensor CustomLossImpl::forward(const torch::Tensor& X,
        const torch::Tensor& E, const torch::Tensor& phi,
        const torch::Tensor& y_obs, const torch::Tensor& sigma) {

    const int n = static_cast<int>(X.size(0));

    torch::Tensor chi2 = torch::zeros({}, kF64);

    for (int i = 0; i < n; ++i) {

        const double xB = X[i][0].item<double>();
        const double t = X[i][1].item<double>();
        const double Q2 = X[i][2].item<double>();
        const double Eb = E[i].item<double>();
        const double ph = phi[i].item<double>();

        // Full kinematics, kept general for phi-dependent observables.
        // (For the sin1phi moment, phi is integrated out by the observable.)
        PARTONS::DVCSObservableKinematic kin(xB, t, Q2, Eb, ph);

        // Observable through the differentiable chain — grad-connected to the NN.
        torch::Tensor pred =
                m_pServiceTorch->computeSingleKinematicTorch(kin, m_pObsTorch);

        torch::Tensor resid =
                (pred - y_obs[i].to(kF64)) / sigma[i].to(kF64);
        chi2 = chi2 + resid * resid;
    }

    return m_normalize ? chi2 / n : chi2;
}