//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluIntSin2PhiTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAluIntSin2PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluIntSin2PhiTorch("DVCSAluIntSin2PhiTorch"));

DVCSAluIntSin2PhiTorch::DVCSAluIntSin2PhiTorch(const std::string& className)
        : PARTONS::DVCSAluIntSin2Phi(className), DVCSObservableTorch(),
          MathIntegratorModuleTorch() {
    // Same fixed-order Gauss-Legendre rule as the other moment leaves; see
    // DVCSAluMinusSin1PhiTorch for why the order is 40.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAluIntSin2PhiTorch::DVCSAluIntSin2PhiTorch(const DVCSAluIntSin2PhiTorch& other)
        : PARTONS::DVCSAluIntSin2Phi(other), DVCSObservableTorch(other),
          MathIntegratorModuleTorch(other) {
}

DVCSAluIntSin2PhiTorch::~DVCSAluIntSin2PhiTorch() {
}

DVCSAluIntSin2PhiTorch* DVCSAluIntSin2PhiTorch::clone() const {
    return new DVCSAluIntSin2PhiTorch(*this);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluIntSin2PhiTorch::computeTensorImplBatch(
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

    // A_LU(phi) * sin(2phi), batched over N points x the shared GL nodes.
    // The pointwise layer is a static of the torch pointwise class, which this
    // leaf no longer derives from (it derives from PARTONS::DVCSAluIntSin2Phi).
    DVCSProcessModuleTorch& proc =
            DVCSProcessModuleTorch::from(m_pProcessModule, getClassName());
    auto integrand = [&proc, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return DVCSAluIntTorch::aLUTensorBatch(proc, xB, t, Q2, E, phi).getValue() * torch::sin(2. * phi);
    };

    return PARTONS::PhysicalType<torch::Tensor>(
            integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
                    / PARTONS::Constant::PI, PARTONS::PhysicalUnit::NONE);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluIntSin2PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<double> DVCSAluIntSin2PhiTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    if (!DVCSProcessModuleTorch::tryFrom(m_pProcessModule))
        return PARTONS::DVCSAluIntSin2Phi::computeObservable(kinematic, gpdType);

    torch::NoGradGuard no_grad;
    DVCSObservableResultTorch r = computeTensor(kinematic);
    return PARTONS::PhysicalType<double>(r.getTensor().item<double>(),
            r.getUnit());
}
