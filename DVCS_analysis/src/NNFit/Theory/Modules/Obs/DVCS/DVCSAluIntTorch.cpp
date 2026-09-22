//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluIntTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
// Complete types needed here: the source of the dynamic_cast below, and
// PhysicalUnit for the scalar wrapper.
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalUnit.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAluIntTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluIntTorch("DVCSAluIntTorch"));

DVCSAluIntTorch::DVCSAluIntTorch(const std::string& className)
        : PARTONS::DVCSAluInt(className), DVCSObservableTorch() {
}

DVCSAluIntTorch::DVCSAluIntTorch(const DVCSAluIntTorch& other)
        : PARTONS::DVCSAluInt(other), DVCSObservableTorch(other) {
}

DVCSAluIntTorch::~DVCSAluIntTorch() {
}

DVCSAluIntTorch* DVCSAluIntTorch::clone() const {
    return new DVCSAluIntTorch(*this);
}

DVCSProcessModuleTorch* DVCSAluIntTorch::torchProcessModule() {
    DVCSProcessModuleTorch* pProc =
            dynamic_cast<DVCSProcessModuleTorch*>(m_pProcessModule);
    if (!pProc) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSProcessModuleTorch process module.");
    }
    return pProc;
}

torch::Tensor DVCSAluIntTorch::aLUTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();

    // Prepare once (kinematics + one batched NN forward), then assemble only
    // the cross sections this variant's formula needs.
    pProc->prepareTensorBatch(xB, t, Q2, E);
    return asymmetryTensorBatch(*pProc, phi);
}

torch::Tensor DVCSAluIntTorch::asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
        const torch::Tensor& phi) {

    // Differencing over beam charge isolates the interference term (odd in
    // charge) in the numerator, while the denominator keeps the charge sum.
    torch::Tensor sPP = proc.crossSectionTensorBatch(+1., +1., phi);
    torch::Tensor sPM = proc.crossSectionTensorBatch(+1., -1., phi);
    torch::Tensor sMP = proc.crossSectionTensorBatch(-1., +1., phi);
    torch::Tensor sMM = proc.crossSectionTensorBatch(-1., -1., phi);

    torch::Tensor numerator   = (sPP - sPM) - (sMP - sMM);
    torch::Tensor denominator = (sPP + sPM) + (sMP + sMM);

    return numerator / denominator;
}

torch::Tensor DVCSAluIntTorch::computeTensorImplBatch(
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

    return aLUTensorBatch(xB, t, Q2, E, phi).squeeze(1);
}

torch::Tensor DVCSAluIntTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}

PARTONS::PhysicalType<double> DVCSAluIntTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    torch::NoGradGuard no_grad;
    double value = computeTensor(kinematic).item<double>();
    return PARTONS::PhysicalType<double>(value, PARTONS::PhysicalUnit::NONE);
}
