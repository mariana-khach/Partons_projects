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

    return aLUTensorBatch(xB, t, Q2, E, phi).squeeze(1); // [N,1] -> [N]
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