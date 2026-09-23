//
// Created by Mariana Khachatryan on 9/23/26.
//

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

torch::Tensor DVCSCrossSectionUUMinusTorch::crossSectionNbTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();
    pProc->prepareTensorBatch(xB, t, Q2, E);

    // Both beam helicities at charge -1, summed as PARTONS::DVCSCrossSectionUUMinus does.
    torch::Tensor A = pProc->crossSectionTensorBatch(+1., -1., phi,
            PARTONS::VCSSubProcessType::ALL);
    torch::Tensor B = pProc->crossSectionTensorBatch(-1., -1., phi,
            PARTONS::VCSSubProcessType::ALL);

    // The beam is UNPOLARIZED, so the helicities are AVERAGED (/2) -- where an
    // asymmetry would divide by their sum. The 2pi integrates out the
    // transversely-polarized-target azimuth, turning the 5-fold differential
    // cross section into the 4-fold one PARTONS reports.
    torch::Tensor sigma = (A + B) / 2.;
    sigma = sigma * (2. * PARTONS::Constant::PI);

    // GeV^-2 -> nb, the conversion makeSameUnitAs(PhysicalUnit::NB) performs.
    return sigma * PARTONS::Constant::CONV_GEVm2_TO_NBARN;
}

torch::Tensor DVCSCrossSectionUUMinusTorch::computeTensorImplBatch(
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

    return crossSectionNbTensorBatch(xB, t, Q2, E, phi).squeeze(1);
}

torch::Tensor DVCSCrossSectionUUMinusTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}

PARTONS::PhysicalType<double> DVCSCrossSectionUUMinusTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    torch::NoGradGuard no_grad;
    double value = computeTensor(kinematic).item<double>();
    return PARTONS::PhysicalType<double>(value, PARTONS::PhysicalUnit::NB);
}
