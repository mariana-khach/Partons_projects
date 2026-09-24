//
// Created by Mariana Khachatryan on 9/23/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUMinusTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/beans/process/VCSSubProcessType.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalUnit.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSCrossSectionUUMinusTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCrossSectionUUMinusTorch("DVCSCrossSectionUUMinusTorch"));

DVCSCrossSectionUUMinusTorch::DVCSCrossSectionUUMinusTorch(const std::string& className)
        : PARTONS::DVCSCrossSectionUUMinus(className), DVCSObservableTorch() {
}

DVCSCrossSectionUUMinusTorch::DVCSCrossSectionUUMinusTorch(const DVCSCrossSectionUUMinusTorch& other)
        : PARTONS::DVCSCrossSectionUUMinus(other), DVCSObservableTorch(other) {
}

DVCSCrossSectionUUMinusTorch::~DVCSCrossSectionUUMinusTorch() {
}

DVCSCrossSectionUUMinusTorch* DVCSCrossSectionUUMinusTorch::clone() const {
    return new DVCSCrossSectionUUMinusTorch(*this);
}

DVCSProcessModuleTorch* DVCSCrossSectionUUMinusTorch::torchProcessModule() {
    DVCSProcessModuleTorch* pProc =
            dynamic_cast<DVCSProcessModuleTorch*>(m_pProcessModule);
    if (!pProc) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSProcessModuleTorch process module.");
    }
    return pProc;
}

PARTONS::PhysicalType<torch::Tensor> DVCSCrossSectionUUMinusTorch::crossSectionNbTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();
    pProc->prepareTensorBatch(xB, t, Q2, E);

    // Both beam helicities at charge -1, summed as PARTONS::DVCSCrossSectionUUMinus does.
    PARTONS::PhysicalType<torch::Tensor> A = pProc->crossSectionTensorBatch(+1., -1., phi,
            PARTONS::VCSSubProcessType::ALL);
    PARTONS::PhysicalType<torch::Tensor> B = pProc->crossSectionTensorBatch(-1., -1., phi,
            PARTONS::VCSSubProcessType::ALL);

    // The beam is UNPOLARIZED, so the helicities are AVERAGED (/2) -- where an
    // asymmetry would divide by their sum. The 2pi integrates out the
    // transversely-polarized-target azimuth, turning the 5-fold differential
    // cross section into the 4-fold one PARTONS reports.
    // The beam is UNPOLARIZED, so the helicities are AVERAGED (/2) -- where an
    // asymmetry would divide by their sum. The 2pi integrates out the
    // transversely-polarized-target azimuth, turning the 5-fold differential
    // cross section into the 4-fold one PARTONS reports. (/2 then x2pi = xPI.)
    // The + is unit-checked: two GeV^-2 terms.
    PARTONS::PhysicalType<torch::Tensor> sum = A + B;
    PARTONS::PhysicalType<torch::Tensor> sigma(
            sum.getValue() * PARTONS::Constant::PI, sum.getUnit());

    // GeV^-2 -> nb. Now that the value is unit-tagged this is the SAME call
    // PARTONS makes, rather than a hand-copied CONV_GEVm2_TO_NBARN: the
    // conversion is declared, not transcribed, and cannot drift from PARTONS'.
    return sigma.makeSameUnitAs(PARTONS::PhysicalUnit::NB);
}

PARTONS::PhysicalType<torch::Tensor> DVCSCrossSectionUUMinusTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {

    const size_t N = kinematics.size();
    std::vector<double> xBVec(N), tVec(N), Q2Vec(N), EVec(N), phiVec(N);
    for (size_t i = 0; i < N; ++i) {
        const PARTONS::DVCSObservableKinematic& kin = kinematics[i];
        xBVec[i]  = kin.getXB().getValue();
        tVec[i]   = kin.getT().getValue();
        Q2Vec[i]  = kin.getQ2().getValue();
        EVec[i]   = kin.getE().getValue();
        phiVec[i] = kin.getPhi().getValue();
    }
    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor xB  = torch::tensor(xBVec, f64);
    torch::Tensor t   = torch::tensor(tVec, f64);
    torch::Tensor Q2  = torch::tensor(Q2Vec, f64);
    torch::Tensor E   = torch::tensor(EVec, f64);
    torch::Tensor phi = torch::tensor(phiVec, f64).unsqueeze(1); // [N] -> [N,1]

    PARTONS::PhysicalType<torch::Tensor> r =
            crossSectionNbTensorBatch(xB, t, Q2, E, phi);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue().squeeze(1),
            r.getUnit()); // [N,1] -> [N]
}

PARTONS::PhysicalType<torch::Tensor> DVCSCrossSectionUUMinusTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<double> DVCSCrossSectionUUMinusTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    torch::NoGradGuard no_grad;
    DVCSObservableResultTorch r =
            computeTensor(kinematic);
    // The unit is taken FROM the tensor result rather than hardcoded here, so
    // the two paths cannot disagree about what this observable returns.
    return PARTONS::PhysicalType<double>(r.getTensor().item<double>(),
            r.getUnit());
}
