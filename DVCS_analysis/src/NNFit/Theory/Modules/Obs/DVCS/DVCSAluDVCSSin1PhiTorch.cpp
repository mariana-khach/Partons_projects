//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluDVCSSin1PhiTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAluDVCSSin1PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluDVCSSin1PhiTorch("DVCSAluDVCSSin1PhiTorch"));

DVCSAluDVCSSin1PhiTorch::DVCSAluDVCSSin1PhiTorch(const std::string& className)
        : PARTONS::DVCSAluDVCSSin1Phi(className), DVCSObservableTorch(),
          MathIntegratorModuleTorch() {
    // Same fixed-order Gauss-Legendre rule as the other moment leaves; see
    // DVCSAluMinusSin1PhiTorch for why the order is 40.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAluDVCSSin1PhiTorch::DVCSAluDVCSSin1PhiTorch(const DVCSAluDVCSSin1PhiTorch& other)
        : PARTONS::DVCSAluDVCSSin1Phi(other), DVCSObservableTorch(other),
          MathIntegratorModuleTorch(other) {
}

DVCSAluDVCSSin1PhiTorch::~DVCSAluDVCSSin1PhiTorch() {
}

DVCSAluDVCSSin1PhiTorch* DVCSAluDVCSSin1PhiTorch::clone() const {
    return new DVCSAluDVCSSin1PhiTorch(*this);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluDVCSSin1PhiTorch::computeTensorImplBatch(
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

    // A_LU(phi) * sin(1phi), batched over N points x the shared GL nodes.
    // The pointwise layer is a static of the torch pointwise class, which this
    // leaf no longer derives from (it derives from PARTONS::DVCSAluDVCSSin1Phi).
    DVCSProcessModuleTorch& proc =
            DVCSProcessModuleTorch::from(m_pProcessModule, getClassName());
    auto integrand = [&proc, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return DVCSAluDVCSTorch::aLUTensorBatch(proc, xB, t, Q2, E, phi).getValue() * torch::sin(1. * phi);
    };

    return PARTONS::PhysicalType<torch::Tensor>(
            integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
                    / PARTONS::Constant::PI, PARTONS::PhysicalUnit::NONE);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluDVCSSin1PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<double> DVCSAluDVCSSin1PhiTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    if (!DVCSProcessModuleTorch::tryFrom(m_pProcessModule))
        return PARTONS::DVCSAluDVCSSin1Phi::computeObservable(kinematic, gpdType);

    torch::NoGradGuard no_grad;
    DVCSObservableResultTorch r = computeTensor(kinematic);
    return PARTONS::PhysicalType<double>(r.getTensor().item<double>(),
            r.getUnit());
}
