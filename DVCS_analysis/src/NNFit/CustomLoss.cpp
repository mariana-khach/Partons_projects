//
// Created by Mariana Khachatryan on 6/17/26.
//

#include "../../include/NNFit/CustomLoss.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
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
        const std::vector<std::string>& outputLayer,
        const std::string& observableName, const torch::Tensor& xMin,
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

    // The observable comes from the DATA FILE's header, not from this file.
    // The tensor leaf is the PARTONS class name + "Torch".
    const std::string torchClassName = observableName + "Torch";
    DVCSObservable* pObs = 0;
    try {
        pObs = Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                torchClassName);
    } catch (const ElemUtils::CustomException&) {
        // PARTONS' own message names only torchClassName -- a string the user
        // never typed, since the "Torch" suffix is appended here. Say where the
        // name came from, and separate the two causes: a misspelled header, or
        // an observable with no tensor twin (most of them: the polarized-target
        // sector is not ported).
        throw ElemUtils::CustomException("CustomLossImpl", __func__,
                ElemUtils::Formatter()
                        << "Data file header names observable '"
                        << observableName << "', but no tensor twin '"
                        << torchClassName << "' is registered. Either the name "
                        << "is misspelled, or that observable is not ported to "
                        << "the torch chain.");
    }

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

torch::Tensor CustomLossImpl::forward(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics,
        const torch::Tensor& y_obs, const torch::Tensor& sigma) {

    const int n = static_cast<int>(kinematics.size());

    // Observable through the differentiable batched chain — grad-connected to
    // the NN, one call for all N points (no per-row bean construction here;
    // the list was already built once by the caller -- see fit_once()).
    torch::Tensor pred = m_pServiceTorch->computeManyKinematicTorch(kinematics,
            m_pObsTorch).getTensor();

    torch::Tensor resid = (pred - y_obs.to(kF64)) / sigma.to(kF64);
    torch::Tensor chi2 = (resid * resid).sum();

    return m_normalize ? chi2 / n : chi2;
}