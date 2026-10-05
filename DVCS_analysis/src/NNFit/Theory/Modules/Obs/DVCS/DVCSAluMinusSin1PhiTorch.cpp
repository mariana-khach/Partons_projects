//
// Created by Mariana Khachatryan on 6/15/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.h"

#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAluMinusSin1PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluMinusSin1PhiTorch("DVCSAluMinusSin1PhiTorch"));

DVCSAluMinusSin1PhiTorch::DVCSAluMinusSin1PhiTorch(const std::string& className) :
        PARTONS::DVCSAluMinusSin1Phi(className), DVCSObservableTorch(),
        MathIntegratorModuleTorch() {
    // Fixed-order Gauss-Legendre over phi in [0, 2pi]. The A_LU^{sin1phi}
    // integrand is smooth and 2pi-periodic, so a fixed rule is one batched
    // integrand evaluation (vs DEXP's adaptive multi-level, which
    // integrateTorchBatch does not support at all).
    //
    // 40 nodes. Measured against the scalar path's adaptive DEXP over every
    // kinematic point of the dataset, by observ_calc_scalar_cff():
    //
    //   order   sin(1phi)   sin(2phi)
    //   GL-10    4.2e-4        --
    //   GL-20    1.2e-8      4.8e-7
    //   GL-40    1.8e-13     7.0e-12     <- here
    //   GL-80    1.7e-13     6.8e-12     (no further gain; beyond the floor an
    //                                     observable can get WORSE, e.g.
    //                                     AluIntSin2Phi 6.5e-12 -> 1.9e-10)
    //
    // Two reasons for 40 over 20. A higher harmonic is less well resolved at a
    // given order -- sin(2phi) sits ~40x looser than sin(1phi) at every order
    // -- so 20 is not uniformly safe for the family. And at 40 the residual is
    // no longer our quadrature error at all but the scalar side's own DEXP
    // tolerance, which makes the differential test a sharper instrument: a
    // transcription bug below ~5e-7 would hide inside the GL-20 residual, while
    // at GL-40 the detection threshold is ~1e-11. That matters with 50 more
    // observables to port.
    //
    // The cost is nil: batched time tracks operation COUNT, not element count,
    // and M enters the [N,M] tensors as elements. Doubling 10 -> 20 measured
    // +0.06% per epoch; see the 2026-09-22 notes for the 20 -> 40 measurement.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSAluMinusSin1PhiTorch::DVCSAluMinusSin1PhiTorch(
        const DVCSAluMinusSin1PhiTorch& other) :
        PARTONS::DVCSAluMinusSin1Phi(other), DVCSObservableTorch(other),
        MathIntegratorModuleTorch(other) {
}

DVCSAluMinusSin1PhiTorch::~DVCSAluMinusSin1PhiTorch() {
}

DVCSAluMinusSin1PhiTorch* DVCSAluMinusSin1PhiTorch::clone() const {
    return new DVCSAluMinusSin1PhiTorch(*this);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusSin1PhiTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    // Thin N=1 wrapper around computeTensorImplBatch(): wrap the single
    // kinematic into a one-element List<K> (cheap -- the bean travels as-is,
    // no field extraction) and delegate.
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusSin1PhiTorch::computeTensorImplBatch(
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
    // The pointwise layer is a static of the torch pointwise class, which this
    // leaf no longer derives from (it derives from PARTONS::DVCSAluMinusSin1Phi).
    DVCSProcessModuleTorch& proc =
            DVCSProcessModuleTorch::from(m_pProcessModule, getClassName());
    auto integrand = [&proc, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return DVCSAluMinusTorch::aLUTensorBatch(proc, xB, t, Q2, E, phi).getValue() * torch::sin(phi);
    };

    return PARTONS::PhysicalType<torch::Tensor>(
            integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI)
                    / PARTONS::Constant::PI, PARTONS::PhysicalUnit::NONE);
}

PARTONS::PhysicalType<double> DVCSAluMinusSin1PhiTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    if (!DVCSProcessModuleTorch::tryFrom(m_pProcessModule))
        return PARTONS::DVCSAluMinusSin1Phi::computeObservable(kinematic, gpdType);

    torch::NoGradGuard no_grad;
    DVCSObservableResultTorch r = computeTensor(kinematic);
    return PARTONS::PhysicalType<double>(r.getTensor().item<double>(),
            r.getUnit());
}
