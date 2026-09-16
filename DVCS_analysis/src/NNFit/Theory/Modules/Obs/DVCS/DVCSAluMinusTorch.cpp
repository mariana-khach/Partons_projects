//
// Created by Mariana Khachatryan on 6/16/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalUnit.h>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

const unsigned int DVCSAluMinusTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluMinusTorch("DVCSAluMinusTorch"));

DVCSAluMinusTorch::DVCSAluMinusTorch(const std::string& className) :
        PARTONS::DVCSAluMinus(className) {
}

DVCSAluMinusTorch::DVCSAluMinusTorch(const DVCSAluMinusTorch& other) :
        PARTONS::DVCSAluMinus(other) {
}

DVCSAluMinusTorch::~DVCSAluMinusTorch() {
}

DVCSAluMinusTorch* DVCSAluMinusTorch::clone() const {
    return new DVCSAluMinusTorch(*this);
}

DVCSProcessModuleTorch* DVCSAluMinusTorch::torchProcessModule() {
    DVCSProcessModuleTorch* pProc =
            dynamic_cast<DVCSProcessModuleTorch*>(m_pProcessModule);
    if (!pProc) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Tensor path requires a DVCSProcessModuleTorch process module.");
    }
    return pProc;
}

torch::Tensor DVCSAluMinusTorch::aLUTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();

    // Hoist the phi-/helicity-independent setup out of the per-helicity
    // calls: prepare once (N-point kinematics + one batched NN forward), then
    // assemble sigma for each beam helicity from the cached state.
    pProc->prepareTensorBatch(xB, t, Q2, E);
    torch::Tensor sigmaPlus = pProc->crossSectionTensorBatch(+1., -1., phi);
    torch::Tensor sigmaMinus = pProc->crossSectionTensorBatch(-1., -1., phi);

    return (sigmaPlus - sigmaMinus) / (sigmaPlus + sigmaMinus); // [N,M]
}

torch::Tensor DVCSAluMinusTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {
    // See the header doc comment: a correct O(N) implementation needs a
    // per-point-own-phi broadcasting mode aLUTensorBatch() doesn't have
    // (it was built for the shared-quadrature-node Fourier-moment case).
    // Not needed by any current consumer -- DVCSAluMinusSin1PhiTorch
    // overrides this with the real implementation.
    throw ElemUtils::CustomException(getClassName(), __func__,
            "Batched pointwise A_LU is not implemented at this base class; "
            "use a Fourier-moment leaf (e.g. DVCSAluMinusSin1PhiTorch).");
}

torch::Tensor DVCSAluMinusTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    // Thin N=1 wrapper around computeTensorImplBatch(), mirroring
    // DVCSAluMinusSin1PhiTorch::computeTensorImpl(). Not yet implemented at
    // this base class -- computeTensorImplBatch() throws (see its doc comment).
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    return computeTensorImplBatch(list)[0];
}

PARTONS::PhysicalType<double> DVCSAluMinusTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {

    torch::NoGradGuard no_grad;
    double value = computeTensor(kinematic).item<double>();
    return PARTONS::PhysicalType<double>(value, PARTONS::PhysicalUnit::NONE);
}