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
    // Fixed 10-point Gauss-Legendre over phi in [0, 2pi]. The A_LU^{sin1phi}
    // integrand is smooth and 2pi-periodic, so a fixed rule is one batched
    // integrand evaluation (vs DEXP's adaptive multi-level), at the cost of a
    // tiny quadrature difference vs the scalar adaptive path.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 10);
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

    // Thin N=1 wrapper around computeTensorImplBatch(): wrap the single
    // kinematic into a one-element List<K> (cheap -- the bean travels as-is,
    // no field extraction) and delegate.
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}

torch::Tensor DVCSAluMinusSin1PhiTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {

    // Unpack the channel-generic bean list into raw [N] tensors. Each
    // kinematic's own phi is ignored: this observable integrates over the
    // full phi range regardless, same as the (former) computeTensorImpl().
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

    // A_LU(phi) * sin(phi), batched over N data points x the GL-10 quadrature
    // nodes shared by every data point.
    auto integrand = [this, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return aLUTensorBatch(xB, t, Q2, E, phi) * torch::sin(phi);
    };

    return integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
            / PARTONS::Constant::PI;
}