//
// Created by Mariana Khachatryan on 6/16/26.
//

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalUnit.h>

#include "NNFit/Theory/Modules/Processes/DVCS/DVCSProcessModuleTorch.h"

namespace {
const torch::TensorOptions kF64 = torch::TensorOptions().dtype(torch::kFloat64);
} // namespace

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

torch::Tensor DVCSAluMinusTorch::aLUTensor(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();

    // Hoist the phi-/helicity-independent setup (one NN forward + BMJ12
    // kinematics + 72 angular coeffs) out of the per-helicity calls: prepare
    // once, then assemble sigma for each beam helicity from the cached state.
    pProc->prepareTensor(kinematic);
    torch::Tensor sigmaPlus = pProc->crossSectionTensor(+1., -1., phi);
    torch::Tensor sigmaMinus = pProc->crossSectionTensor(-1., -1., phi);

    return (sigmaPlus - sigmaMinus) / (sigmaPlus + sigmaMinus);
}

torch::Tensor DVCSAluMinusTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    // Pointwise asymmetry at the kinematic's stored phi (mirrors the scalar
    // DVCSAluMinus::computeObservable, which evaluates at kinematic.getPhi()).
    const double phi = kinematic.getPhi().getValue();
    torch::Tensor phiT = torch::full({1}, phi, kF64);
    return aLUTensor(kinematic, phiT).squeeze();
}

PARTONS::PhysicalType<double> DVCSAluMinusTorch::computeObservable(
        const PARTONS::DVCSObservableKinematic& kinematic,
        const PARTONS::List<PARTONS::GPDType>& gpdType) {

    torch::NoGradGuard no_grad;
    double value = computeTensor(kinematic).item<double>();
    return PARTONS::PhysicalType<double>(value, PARTONS::PhysicalUnit::NONE);
}