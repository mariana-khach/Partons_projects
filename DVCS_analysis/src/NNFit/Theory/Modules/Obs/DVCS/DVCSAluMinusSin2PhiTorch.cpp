//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusSin2PhiTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>

#include <vector>

const unsigned int DVCSAluMinusSin2PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluMinusSin2PhiTorch("DVCSAluMinusSin2PhiTorch"));

DVCSAluMinusSin2PhiTorch::DVCSAluMinusSin2PhiTorch(const std::string& className)
        : DVCSAluMinusTorch(className), MathIntegratorModuleTorch() {
    // Same fixed-order Gauss-Legendre rule as the other moment leaves; see
    // DVCSAluMinusSin1PhiTorch for why the order is 40.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAluMinusSin2PhiTorch::DVCSAluMinusSin2PhiTorch(const DVCSAluMinusSin2PhiTorch& other)
        : DVCSAluMinusTorch(other), MathIntegratorModuleTorch(other) {
}

DVCSAluMinusSin2PhiTorch::~DVCSAluMinusSin2PhiTorch() {
}

DVCSAluMinusSin2PhiTorch* DVCSAluMinusSin2PhiTorch::clone() const {
    return new DVCSAluMinusSin2PhiTorch(*this);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusSin2PhiTorch::computeTensorImplBatch(
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
    auto integrand = [this, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return aLUTensorBatch(xB, t, Q2, E, phi).getValue() * torch::sin(2. * phi);
    };

    return PARTONS::PhysicalType<torch::Tensor>(
            integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
                    / PARTONS::Constant::PI, PARTONS::PhysicalUnit::NONE);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusSin2PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}
