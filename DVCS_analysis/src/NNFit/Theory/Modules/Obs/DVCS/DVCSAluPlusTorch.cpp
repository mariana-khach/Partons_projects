//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluPlusTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
// Complete types needed here: the source of the dynamic_cast below, and
// PhysicalUnit for the scalar wrapper.
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalUnit.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAluPlusTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluPlusTorch("DVCSAluPlusTorch"));

DVCSAluPlusTorch::DVCSAluPlusTorch(const std::string& className)
        : PARTONS::DVCSAluPlus(className), DVCSObservableTorch() {
}

DVCSAluPlusTorch::DVCSAluPlusTorch(const DVCSAluPlusTorch& other)
        : PARTONS::DVCSAluPlus(other), DVCSObservableTorch(other) {
}

DVCSAluPlusTorch::~DVCSAluPlusTorch() {
}

DVCSAluPlusTorch* DVCSAluPlusTorch::clone() const {
    return new DVCSAluPlusTorch(*this);
}

DVCSProcessModuleTorch* DVCSAluPlusTorch::torchProcessModule() {
    DVCSProcessModuleTorch* pProc =
            dynamic_cast<DVCSProcessModuleTorch*>(m_pProcessModule);
    if (!pProc) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSProcessModuleTorch process module.");
    }
    return pProc;
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluPlusTorch::aLUTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();

    // Prepare once (kinematics + one batched NN forward), then assemble only
    // the cross sections this variant's formula needs.
    pProc->prepareTensorBatch(xB, t, Q2, E);
    return asymmetryTensorBatch(*pProc, phi);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluPlusTorch::asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
        const torch::Tensor& phi) {

    // Beam charge +1 throughout: (s++ - s-+) / (s++ + s-+).
    PARTONS::PhysicalType<torch::Tensor> sPP = proc.crossSectionTensorBatch(+1., +1., phi);
    PARTONS::PhysicalType<torch::Tensor> sMP = proc.crossSectionTensorBatch(-1., +1., phi);

    return (sPP - sMP) / (sPP + sMP);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluPlusTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {

    // Each kinematic at its OWN phi: pass phi as [N,1] rather than the moment
    // leaves' shared [M] grid (see DVCSAluMinusTorch for the broadcasting).
    const size_t N = kinematics.size();
    std::vector<double> xBVec(N), tVec(N), Q2Vec(N), EVec(N), phiVec(N);
    for (size_t i = 0; i < N; ++i) {
        const PARTONS::DVCSObservableKinematic& kin = kinematics[i];
        xBVec[i]  = kin.getXB().getValue();
        tVec[i]   = kin.getT().getValue();
        Q2Vec[i]  = kin.getQ2().getValue();
        EVec[i]   = kin.getE().getValue();
        phiVec[i] = kin.getPhi().getValue();
    }
    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor xB  = torch::tensor(xBVec, f64);
    torch::Tensor t   = torch::tensor(tVec, f64);
    torch::Tensor Q2  = torch::tensor(Q2Vec, f64);
    torch::Tensor E   = torch::tensor(EVec, f64);
    torch::Tensor phi = torch::tensor(phiVec, f64).unsqueeze(1);

    PARTONS::PhysicalType<torch::Tensor> r =
            aLUTensorBatch(xB, t, Q2, E, phi);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue().squeeze(1),
            r.getUnit()); // [N,1] -> [N]
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluPlusTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<double> DVCSAluPlusTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    torch::NoGradGuard no_grad;
    DVCSObservableResultTorch r =
            computeTensor(kinematic);
    // The unit is taken FROM the tensor result rather than hardcoded here, so
    // the two paths cannot disagree about what this observable returns.
    return PARTONS::PhysicalType<double>(r.getTensor().item<double>(),
            r.getUnit());
}
