//
// Created by Mariana Khachatryan on 9/22/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluDVCSSin1PhiTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>

#include <vector>

const unsigned int DVCSAluDVCSSin1PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluDVCSSin1PhiTorch("DVCSAluDVCSSin1PhiTorch"));

DVCSAluDVCSSin1PhiTorch::DVCSAluDVCSSin1PhiTorch(const std::string& className)
        : DVCSAluDVCSTorch(className), MathIntegratorModuleTorch() {
    // Same fixed-order Gauss-Legendre rule as the other moment leaves; see
    // DVCSAluMinusSin1PhiTorch for why the order is 40.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAluDVCSSin1PhiTorch::DVCSAluDVCSSin1PhiTorch(const DVCSAluDVCSSin1PhiTorch& other)
        : DVCSAluDVCSTorch(other), MathIntegratorModuleTorch(other) {
}

DVCSAluDVCSSin1PhiTorch::~DVCSAluDVCSSin1PhiTorch() {
}

DVCSAluDVCSSin1PhiTorch* DVCSAluDVCSSin1PhiTorch::clone() const {
    return new DVCSAluDVCSSin1PhiTorch(*this);
}

torch::Tensor DVCSAluDVCSSin1PhiTorch::computeTensorImplBatch(
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
    auto integrand = [this, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return aLUTensorBatch(xB, t, Q2, E, phi) * torch::sin(1. * phi);
    };

    return integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
            / PARTONS::Constant::PI;
}

torch::Tensor DVCSAluDVCSSin1PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}
