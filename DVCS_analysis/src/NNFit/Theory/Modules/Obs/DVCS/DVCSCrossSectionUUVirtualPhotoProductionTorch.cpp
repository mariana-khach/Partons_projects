//
// Created by Mariana Khachatryan on 9/23/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUVirtualPhotoProductionTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/beans/process/VCSSubProcessType.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalUnit.h>

#include <vector>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSCrossSectionUUVirtualPhotoProductionTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCrossSectionUUVirtualPhotoProductionTorch("DVCSCrossSectionUUVirtualPhotoProductionTorch"));

DVCSCrossSectionUUVirtualPhotoProductionTorch::DVCSCrossSectionUUVirtualPhotoProductionTorch(const std::string& className)
        : PARTONS::DVCSCrossSectionUUVirtualPhotoProduction(className), DVCSObservableTorch() {
}

DVCSCrossSectionUUVirtualPhotoProductionTorch::DVCSCrossSectionUUVirtualPhotoProductionTorch(const DVCSCrossSectionUUVirtualPhotoProductionTorch& other)
        : PARTONS::DVCSCrossSectionUUVirtualPhotoProduction(other), DVCSObservableTorch(other) {
}

DVCSCrossSectionUUVirtualPhotoProductionTorch::~DVCSCrossSectionUUVirtualPhotoProductionTorch() {
}

DVCSCrossSectionUUVirtualPhotoProductionTorch* DVCSCrossSectionUUVirtualPhotoProductionTorch::clone() const {
    return new DVCSCrossSectionUUVirtualPhotoProductionTorch(*this);
}

DVCSProcessModuleTorch* DVCSCrossSectionUUVirtualPhotoProductionTorch::torchProcessModule() {
    DVCSProcessModuleTorch* pProc =
            dynamic_cast<DVCSProcessModuleTorch*>(m_pProcessModule);
    if (!pProc) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSProcessModuleTorch process module.");
    }
    return pProc;
}

torch::Tensor DVCSCrossSectionUUVirtualPhotoProductionTorch::crossSectionNbTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();
    pProc->prepareTensorBatch(xB, t, Q2, E);

    // Both beam helicities at charge -1, summed as PARTONS::DVCSCrossSectionUUVirtualPhotoProduction does.
    torch::Tensor A = pProc->crossSectionTensorBatch(+1., -1., phi,
            PARTONS::VCSSubProcessType::DVCS);
    torch::Tensor B = pProc->crossSectionTensorBatch(-1., -1., phi,
            PARTONS::VCSSubProcessType::DVCS);

    // The beam is UNPOLARIZED, so the helicities are AVERAGED (/2) -- where an
    // asymmetry would divide by their sum. The 2pi integrates out the
    // transversely-polarized-target azimuth, turning the 5-fold differential
    // cross section into the 4-fold one PARTONS reports.
    torch::Tensor sigma = (A + B) / 2.;
    sigma = sigma * (2. * PARTONS::Constant::PI);

    // Virtual-photon flux -- pure kinematics, [N], no grad. Transcribed from
    // PARTONS::DVCSCrossSectionUUVirtualPhotoProduction::getVirtualPhotonFlux().
    // Unsqueezed to [N,1] so it broadcasts against whatever shape phi gave.
    const double M = PARTONS::Constant::PROTON_MASS;
    torch::Tensor nu  = Q2 / (2. * M * xB);
    torch::Tensor y   = nu / E;
    torch::Tensor eps = 2. * xB * M / torch::sqrt(Q2);
    torch::Tensor ey2 = torch::pow(eps * y / 2., 2);
    torch::Tensor e   = (1. - y - ey2) / (1. - y + torch::pow(y, 2) / 2. + ey2);
    torch::Tensor flux = torch::tensor(
            PARTONS::Constant::FINE_STRUCTURE_CONSTANT / (2. * PARTONS::Constant::PI),
            torch::TensorOptions().dtype(torch::kFloat64))
            * torch::pow(y, 2) / (1. - e) * (1. - xB) / (xB * Q2);
    sigma = sigma / flux.unsqueeze(1);

    // GeV^-2 -> nb, the conversion makeSameUnitAs(PhysicalUnit::NB) performs.
    return sigma * PARTONS::Constant::CONV_GEVm2_TO_NBARN;
}

torch::Tensor DVCSCrossSectionUUVirtualPhotoProductionTorch::computeTensorImplBatch(
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

torch::Tensor DVCSCrossSectionUUVirtualPhotoProductionTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}

PARTONS::PhysicalType<double> DVCSCrossSectionUUVirtualPhotoProductionTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {
    torch::NoGradGuard no_grad;
    double value = computeTensor(kinematic).item<double>();
    return PARTONS::PhysicalType<double>(value, PARTONS::PhysicalUnit::NB);
}
