//
// Created by Mariana Khachatryan on 9/23/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUMinusPhiIntegratedTorch.h"

#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSCrossSectionUUMinusPhiIntegratedTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCrossSectionUUMinusPhiIntegratedTorch("DVCSCrossSectionUUMinusPhiIntegratedTorch"));

DVCSCrossSectionUUMinusPhiIntegratedTorch::DVCSCrossSectionUUMinusPhiIntegratedTorch(const std::string& className)
        : PARTONS::DVCSCrossSectionUUMinusPhiIntegrated(className), DVCSObservableTorch(),
          MathIntegratorModuleTorch() {
    // GL-160 -- four times the asymmetry leaves' 40, and this leaf is the ONLY
    // one in the family that needs it. Its integrand is the FULL cross section,
    // so it carries the Bethe-Heitler peak at the interval ends (phi -> 0 and
    // 2pi; measured, the peak is ~5900x the value at phi = pi and is 99.3% BH),
    // which demands far more resolution than a bounded asymmetry.
    //
    // Measured 2026-09-23 per dataset point, against the scalar path:
    //
    //   order          xB=0.25, t=-0.488     points already converged
    //   GL-40               3.0e-5                   ~5e-14
    //   GL-80               3.0e-9                   ~3e-13
    //   GL-160              2.8e-11                  ~3e-12
    //   GL-320              4.1e-12                  ~1e-12
    //
    // Raising the order is a TRADE, not a free win: the already-converged
    // points get ~60x worse. That is not "more nodes, more roundoff" -- it is
    // NumA's rule quality falling off a cliff the moment you leave GL-20/40,
    // the only two orders it tabulates. See setIntegrator() in
    // MathIntegratorModuleTorch.h for the defect and its size (~100x on the
    // weights). 160 is where paying that fixed penalty stops being worth it.
    //
    // Two dataset points do NOT improve at any order (8.1e-6 and 2.5e-4,
    // identical GL-40 through GL-640). Those are the SCALAR side's: a
    // torch-free probe -- PARTONS' own pointwise cross section integrated with
    // GL-40/200/1000 against PARTONS' own DEXP class -- reproduces 2.4842e-04
    // at the worst one, flat in order.
    //
    // Measure per point, never by the max over the dataset: that statistic sat
    // at 2.4848e-04 from GL-40 to GL-640, pinned by the DEXP outlier, while the
    // under-resolved point above was still converging underneath it.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 160);
}

DVCSCrossSectionUUMinusPhiIntegratedTorch::DVCSCrossSectionUUMinusPhiIntegratedTorch(const DVCSCrossSectionUUMinusPhiIntegratedTorch& other)
        : PARTONS::DVCSCrossSectionUUMinusPhiIntegrated(other), DVCSObservableTorch(other),
          MathIntegratorModuleTorch(other) {
}

DVCSCrossSectionUUMinusPhiIntegratedTorch::~DVCSCrossSectionUUMinusPhiIntegratedTorch() {
}

DVCSCrossSectionUUMinusPhiIntegratedTorch* DVCSCrossSectionUUMinusPhiIntegratedTorch::clone() const {
    return new DVCSCrossSectionUUMinusPhiIntegratedTorch(*this);
}

PARTONS::PhysicalType<torch::Tensor> DVCSCrossSectionUUMinusPhiIntegratedTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {

    // Each kinematic's own phi is ignored: this observable integrates over the
    // full phi range regardless, as the moment leaves do.
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

    // The pointwise layer is a static of the torch pointwise class, which this
    // leaf no longer derives from (it derives from PARTONS::DVCSCrossSectionUUMinusPhiIntegrated).
    DVCSProcessModuleTorch& proc =
            DVCSProcessModuleTorch::from(m_pProcessModule, getClassName());
    auto integrand = [&proc, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return DVCSCrossSectionUUMinusTorch::crossSectionNbTensorBatch(proc, xB, t, Q2, E, phi).getValue();
    };

    // No normalization -- PARTONS::DVCSCrossSectionUUMinusPhiIntegrated returns the bare
    // integral, unlike the Fourier moments which divide by pi or 2pi.
    return PARTONS::PhysicalType<torch::Tensor>(
            integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI),
            PARTONS::PhysicalUnit::NB);   // a phi-integrated CROSS SECTION
}

PARTONS::PhysicalType<torch::Tensor> DVCSCrossSectionUUMinusPhiIntegratedTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<double> DVCSCrossSectionUUMinusPhiIntegratedTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    if (!DVCSProcessModuleTorch::tryFrom(m_pProcessModule))
        return PARTONS::DVCSCrossSectionUUMinusPhiIntegrated::computeObservable(kinematic, gpdType);

    torch::NoGradGuard no_grad;
    DVCSObservableResultTorch r = computeTensor(kinematic);
    return PARTONS::PhysicalType<double>(r.getTensor().item<double>(),
            r.getUnit());
}
