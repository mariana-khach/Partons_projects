//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAcCos3PhiTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>

#include <vector>

const unsigned int DVCSAcCos3PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAcCos3PhiTorch("DVCSAcCos3PhiTorch"));

DVCSAcCos3PhiTorch::DVCSAcCos3PhiTorch(const std::string& className)
        : DVCSAcTorch(className), MathIntegratorModuleTorch() {
    // Same fixed-order rule as the A_LU moment leaves; see
    // DVCSAluMinusSin1PhiTorch for why the order is 40.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAcCos3PhiTorch::DVCSAcCos3PhiTorch(const DVCSAcCos3PhiTorch& other)
        : DVCSAcTorch(other), MathIntegratorModuleTorch(other) {
}

DVCSAcCos3PhiTorch::~DVCSAcCos3PhiTorch() {
}

DVCSAcCos3PhiTorch* DVCSAcCos3PhiTorch::clone() const {
    return new DVCSAcCos3PhiTorch(*this);
}

torch::Tensor DVCSAcCos3PhiTorch::computeTensorImplBatch(
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
    auto integrand = [this, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return aCTensorBatch(xB, t, Q2, E, phi) * torch::cos(3. * phi);
    };

    return integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
            / PARTONS::Constant::PI;
}

torch::Tensor DVCSAcCos3PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}
