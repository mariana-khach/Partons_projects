//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAcCos3PhiTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAcCos3PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAcCos3PhiTorch("DVCSAcCos3PhiTorch"));

DVCSAcCos3PhiTorch::DVCSAcCos3PhiTorch(const std::string& className)
        : PARTONS::DVCSAcCos3Phi(className), DVCSObservableTorch(),
          MathIntegratorModuleTorch() {
    // Same fixed-order rule as the A_LU moment leaves; see
    // DVCSAluMinusSin1PhiTorch for why the order is 40.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAcCos3PhiTorch::DVCSAcCos3PhiTorch(const DVCSAcCos3PhiTorch& other)
        : PARTONS::DVCSAcCos3Phi(other), DVCSObservableTorch(other),
          MathIntegratorModuleTorch(other) {
}

DVCSAcCos3PhiTorch::~DVCSAcCos3PhiTorch() {
}

DVCSAcCos3PhiTorch* DVCSAcCos3PhiTorch::clone() const {
    return new DVCSAcCos3PhiTorch(*this);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAcCos3PhiTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {

    const size_t N = kinematics.size();
    std::vector<double> xBVec(N), tVec(N), Q2Vec(N), EVec(N);
    for (size_t i = 0; i < N; ++i) {
        const PARTONS::DVCSObservableKinematic& kin = kinematics[i];
        xBVec[i] = kin.getXB().getValue();
        tVec[i]  = kin.getT().getValue();
        Q2Vec[i] = kin.getQ2().getValue();
        EVec[i]  = kin.getE().getValue();
    }
    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor xB = torch::tensor(xBVec, f64);
    torch::Tensor t  = torch::tensor(tVec, f64);
    torch::Tensor Q2 = torch::tensor(Q2Vec, f64);
    torch::Tensor E  = torch::tensor(EVec, f64);

    // Weight cos(3phi), normalization 1/pi -- as in PARTONS::DVCSAcCos3Phi.
    // The pointwise layer is a static of the torch pointwise class, which this
    // leaf no longer derives from (it derives from PARTONS::DVCSAcCos3Phi).
    DVCSProcessModuleTorch& proc =
            DVCSProcessModuleTorch::from(m_pProcessModule, getClassName());
    auto integrand = [&proc, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return DVCSAcTorch::aCTensorBatch(proc, xB, t, Q2, E, phi).getValue() * torch::cos(3. * phi);
    };

    return PARTONS::PhysicalType<torch::Tensor>(
            integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
                    / PARTONS::Constant::PI, PARTONS::PhysicalUnit::NONE);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAcCos3PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<double> DVCSAcCos3PhiTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    if (!DVCSProcessModuleTorch::tryFrom(m_pProcessModule))
        return PARTONS::DVCSAcCos3Phi::computeObservable(kinematic, gpdType);

    torch::NoGradGuard no_grad;
    DVCSObservableResultTorch r = computeTensor(kinematic);
    return PARTONS::PhysicalType<double>(r.getTensor().item<double>(),
            r.getUnit());
}
