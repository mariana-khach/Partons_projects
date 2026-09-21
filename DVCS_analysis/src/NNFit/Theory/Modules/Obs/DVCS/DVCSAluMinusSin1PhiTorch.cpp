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
    // Fixed-order Gauss-Legendre over phi in [0, 2pi]. The A_LU^{sin1phi}
    // integrand is smooth and 2pi-periodic, so a fixed rule is one batched
    // integrand evaluation (vs DEXP's adaptive multi-level, which
    // integrateTorchBatch does not support at all).
    //
    // 20 nodes, not 10. The dataset scan in observ_calc_scalar_cff() measured
    // GL-10 against the scalar path's adaptive DEXP over every kinematic point
    // of the input file and found up to 4.2e-4 relative error -- 100x worse
    // than the single-point check of 2026-06-22 suggested. Order convergence
    // (GL-10 4.2e-4, GL-20 1.2e-8, GL-40 1.8e-13, GL-80 at the double-precision
    // floor) identifies that residual as pure quadrature error, so the fix is
    // simply more nodes. GL-20 buys ~4 orders of magnitude; the remaining gap
    // is far below anything the fit can see.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 20);
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