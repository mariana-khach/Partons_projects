//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAcCos0PhiTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>

#include <vector>

const unsigned int DVCSAcCos0PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAcCos0PhiTorch("DVCSAcCos0PhiTorch"));

DVCSAcCos0PhiTorch::DVCSAcCos0PhiTorch(const std::string& className)
        : DVCSAcTorch(className), MathIntegratorModuleTorch() {
    // Same fixed-order rule as the A_LU moment leaves; see
    // DVCSAluMinusSin1PhiTorch for why the order is 40.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAcCos0PhiTorch::DVCSAcCos0PhiTorch(const DVCSAcCos0PhiTorch& other)
        : DVCSAcTorch(other), MathIntegratorModuleTorch(other) {
}

DVCSAcCos0PhiTorch::~DVCSAcCos0PhiTorch() {
}

DVCSAcCos0PhiTorch* DVCSAcCos0PhiTorch::clone() const {
    return new DVCSAcCos0PhiTorch(*this);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAcCos0PhiTorch::computeTensorImplBatch(
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

    // n = 0 is the plain average: no weight, and the 1/(2 pi) normalization
    // that distinguishes the zeroth Fourier coefficient from the rest.
    auto integrand = [this, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return aCTensorBatch(xB, t, Q2, E, phi).getValue();
    };

    return PARTONS::PhysicalType<torch::Tensor>(
            integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
                    / (2. * PARTONS::Constant::PI), PARTONS::PhysicalUnit::NONE);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAcCos0PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}
