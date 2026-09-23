//
// Created by Mariana Khachatryan on 9/23/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch.h"

#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/FundamentalPhysicalConstants.h>

#include <vector>

const unsigned int DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch("DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch"));

DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch(const std::string& className)
        : DVCSCrossSectionUUDVCSSubProcTorch(className), MathIntegratorModuleTorch() {
    // GL-40 -- the same order the asymmetry leaves use, but measured here, not
    // inherited. This integrand is the pure-DVCS (VCS) sub-process alone, with
    // no Bethe-Heitler term, so it has none of the endpoint peak that forces
    // DVCSCrossSectionUUMinusPhiIntegratedTorch up to GL-160. (Measured: the
    // DVCS sub-process varies only ~35% across the whole phi range, against a
    // factor ~5900 for the full cross section.)
    //
    // Per dataset point, 2026-09-23, against the scalar path:
    //
    //   order     already-converged points     the rest
    //   GL-40              3.7e-15             flat at 1.6e-7 / 8.7e-8 / ...
    //   GL-80              1.6e-13             unchanged
    //   GL-160             8.6e-13             unchanged
    //   GL-320             7.3e-13             unchanged
    //
    // Raising the order only makes it WORSE. The residuals that remain are flat
    // at every order, so they are the scalar side's DEXP and no rule of ours
    // will move them -- while our own error jumps ~40x at 40 -> 80 and stays
    // there. That jump is not roundoff from extra nodes: 40 is the largest
    // order NumA tabulates, and every order above it uses a Newton solver whose
    // weights are ~100x worse (see setIntegrator() in
    // MathIntegratorModuleTorch.h). 40 is already the right answer here.
    MathIntegratorModuleTorch::setIntegrator(NumA::IntegratorType1D::GL, 40);
}

DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch(const DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch& other)
        : DVCSCrossSectionUUDVCSSubProcTorch(other), MathIntegratorModuleTorch(other) {
}

DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::~DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch() {
}

DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch* DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::clone() const {
    return new DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch(*this);
}

torch::Tensor DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::computeTensorImplBatch(
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

    auto integrand = [this, &xB, &t, &Q2, &E](const torch::Tensor& phi) -> torch::Tensor {
        return crossSectionNbTensorBatch(xB, t, Q2, E, phi);
    };

    // No normalization -- PARTONS::DVCSCrossSectionUUDVCSSubProcPhiIntegrated returns the bare
    // integral, unlike the Fourier moments which divide by pi or 2pi.
    return integrateTorchBatch(integrand, 0., 2. * PARTONS::Constant::PI);
}

torch::Tensor DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}
