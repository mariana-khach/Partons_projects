//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAcTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalUnit.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAcTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAcTorch("DVCSAcTorch"));

DVCSAcTorch::DVCSAcTorch(const std::string& className)
        : PARTONS::DVCSAc(className), DVCSObservableTorch() {
}

DVCSAcTorch::DVCSAcTorch(const DVCSAcTorch& other)
        : PARTONS::DVCSAc(other), DVCSObservableTorch(other) {
}

DVCSAcTorch::~DVCSAcTorch() {
}

DVCSAcTorch* DVCSAcTorch::clone() const {
    return new DVCSAcTorch(*this);
}

DVCSProcessModuleTorch* DVCSAcTorch::torchProcessModule() {
    DVCSProcessModuleTorch* pProc =
            dynamic_cast<DVCSProcessModuleTorch*>(m_pProcessModule);
    if (!pProc) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSProcessModuleTorch process module.");
    }
    return pProc;
}

torch::Tensor DVCSAcTorch::aCTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();
    pProc->prepareTensorBatch(xB, t, Q2, E);
    return asymmetryTensorBatch(*pProc, phi);
}

torch::Tensor DVCSAcTorch::asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
        const torch::Tensor& phi) {

    // Sum over beam helicity for each charge (unpolarized beam), then the
    // charge asymmetry. The helicity sum cancels the beam-spin-odd part; the
    // charge difference then isolates the interference term.
    torch::Tensor sPP = proc.crossSectionTensorBatch(+1., +1., phi);
    torch::Tensor sMP = proc.crossSectionTensorBatch(-1., +1., phi);
    torch::Tensor sPM = proc.crossSectionTensorBatch(+1., -1., phi);
    torch::Tensor sMM = proc.crossSectionTensorBatch(-1., -1., phi);

    torch::Tensor plusCharge  = sPP + sMP;
    torch::Tensor minusCharge = sPM + sMM;

    return (plusCharge - minusCharge) / (plusCharge + minusCharge);
}

torch::Tensor DVCSAcTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {

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

    return aCTensorBatch(xB, t, Q2, E, phi).squeeze(1);
}

torch::Tensor DVCSAcTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}

PARTONS::PhysicalType<double> DVCSAcTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    torch::NoGradGuard no_grad;
    double value = computeTensor(kinematic).item<double>();
    return PARTONS::PhysicalType<double>(value, PARTONS::PhysicalUnit::NONE);
}
