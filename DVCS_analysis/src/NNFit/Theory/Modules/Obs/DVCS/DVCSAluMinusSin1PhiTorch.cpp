//
// Created by Mariana Khachatryan on 6/15/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.h"

#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>

const unsigned int DVCSAluMinusSin1PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluMinusSin1PhiTorch("DVCSAluMinusSin1PhiTorch"));

DVCSAluMinusSin1PhiTorch::DVCSAluMinusSin1PhiTorch(const std::string& className) :
        DVCSAluMinusTorch(className), MathIntegratorModuleTorch() {
    // Same integrator as the scalar DVCSAluMinusSin1Phi.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::DEXP);
}

DVCSAluMinusSin1PhiTorch::DVCSAluMinusSin1PhiTorch(
        const DVCSAluMinusSin1PhiTorch& other) :
        DVCSAluMinusTorch(other), MathIntegratorModuleTorch(other) {
}

DVCSAluMinusSin1PhiTorch::~DVCSAluMinusSin1PhiTorch() {
}

DVCSAluMinusSin1PhiTorch* DVCSAluMinusSin1PhiTorch::clone() const {
    return new DVCSAluMinusSin1PhiTorch(*this);
}

torch::Tensor DVCSAluMinusSin1PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    // A_LU(phi) * sin(phi), batched over the quadrature nodes. aLUTensor() is the
    // reusable pointwise asymmetry inherited from DVCSAluMinusTorch.
    auto integrand = [this, &kinematic](const torch::Tensor& phi) -> torch::Tensor {
        return aLUTensor(kinematic, phi) * torch::sin(phi);
    };

    return integrateTorch(integrand, 0., 2. * PARTONS::Constant::PI)
            / PARTONS::Constant::PI;
}