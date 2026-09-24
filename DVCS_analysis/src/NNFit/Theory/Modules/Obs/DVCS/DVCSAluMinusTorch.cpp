//
// Created by Mariana Khachatryan on 6/16/26.
//

#include "NNFit/Theory/Beans/Obs/DVCS/DVCSObservableResultTorch.h"
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

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusTorch::aLUTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E, const torch::Tensor& phi) {

    DVCSProcessModuleTorch* pProc = torchProcessModule();

    // Hoist the phi-/helicity-independent setup out of the per-helicity calls:
    // prepare once (N-point kinematics + one batched NN forward), then let the
    // variant assemble the cross sections its own formula needs from the
    // cached state.
    pProc->prepareTensorBatch(xB, t, Q2, E);
    return asymmetryTensorBatch(*pProc, phi);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusTorch::asymmetryTensorBatch(
        DVCSProcessModuleTorch& proc, const torch::Tensor& phi) {

    // A_LU at beam charge -1: (sigma+- - sigma--) / (sigma+- + sigma--).
    PARTONS::PhysicalType<torch::Tensor> sigmaPlus =
            proc.crossSectionTensorBatch(+1., -1., phi);   // GeV^-2
    PARTONS::PhysicalType<torch::Tensor> sigmaMinus =
            proc.crossSectionTensorBatch(-1., -1., phi);   // GeV^-2

    // PhysicalType's operator/ tags the quotient PhysicalUnit::NONE, so the
    // asymmetry comes out dimensionless by DERIVATION rather than by
    // assertion -- and the +/- above are unit-checked.
    return (sigmaPlus - sigmaMinus) / (sigmaPlus + sigmaMinus);
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusTorch::computeTensorImplBatch(
        const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics) {

    // Pointwise A_LU(phi): each kinematic evaluated at ITS OWN phi, unlike the
    // Fourier-moment leaves which share one quadrature grid across all N points.
    //
    // Both cases run through the same aLUTensorBatch(); only phi's shape picks
    // between them, because every phi-dependent term downstream is built by
    // broadcasting [N] kinematics (unsqueezed to [N,1]) against whatever shape
    // phi has:
    //
    //   phi as [M]    ->  [N,1] x [M]   -> [N,M]   every point at every node
    //   phi as [N,1]  ->  [N,1] x [N,1] -> [N,1]   point i at its own phi_i
    //
    // So this is one batched evaluation over all N points -- no per-point loop,
    // and no O(N^2) diagonal extraction from the shared-grid form. It is in fact
    // cheaper than a moment: the same operation count over M=1 instead of M=20.
    const size_t N = kinematics.size();
    std::vector<double> xBVec(N), tVec(N), Q2Vec(N), EVec(N), phiVec(N);
    for (size_t i = 0; i < N; ++i) {
        const PARTONS::DVCSObservableKinematic& kin = kinematics[i];
        xBVec[i]  = kin.getXB().getValue();
        tVec[i]   = kin.getT().getValue();
        Q2Vec[i]  = kin.getQ2().getValue();
        EVec[i]   = kin.getE().getValue();
        phiVec[i] = kin.getPhi().getValue();   // the moment leaves ignore this
    }
    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor xB  = torch::tensor(xBVec, f64);
    torch::Tensor t   = torch::tensor(tVec, f64);
    torch::Tensor Q2  = torch::tensor(Q2Vec, f64);
    torch::Tensor E   = torch::tensor(EVec, f64);
    torch::Tensor phi = torch::tensor(phiVec, f64).unsqueeze(1); // [N] -> [N,1]

    PARTONS::PhysicalType<torch::Tensor> r = aLUTensorBatch(xB, t, Q2, E, phi);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue().squeeze(1),
            r.getUnit()); // [N,1] -> [N]
}

PARTONS::PhysicalType<torch::Tensor> DVCSAluMinusTorch::computeTensorImpl(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    // Thin N=1 wrapper around computeTensorImplBatch(), mirroring
    // DVCSAluMinusSin1PhiTorch::computeTensorImpl(). Not yet implemented at
    // this base class -- computeTensorImplBatch() throws (see its doc comment).
    PARTONS::List<PARTONS::DVCSObservableKinematic> list;
    list.add(kinematic);
    PARTONS::PhysicalType<torch::Tensor> r = computeTensorImplBatch(list);
    return PARTONS::PhysicalType<torch::Tensor>(r.getValue()[0], r.getUnit());
}

PARTONS::PhysicalType<double> DVCSAluMinusTorch::computeObservable(
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