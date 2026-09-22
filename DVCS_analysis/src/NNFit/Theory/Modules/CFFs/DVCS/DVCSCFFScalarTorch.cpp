//
// Created by Mariana Khachatryan on 9/21/26.
//

#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFScalarTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionResult.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>

#include <complex>
#include <vector>

namespace {

/// The four CFFs the BMJ12 tensor layer consumes, in the order it stores them.
const PARTONS::GPDType::Type kTypes[4] = { PARTONS::GPDType::H,
        PARTONS::GPDType::E, PARTONS::GPDType::Ht, PARTONS::GPDType::Et };

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const unsigned int DVCSCFFScalarTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCFFScalarTorch("DVCSCFFScalarTorch"));

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

DVCSCFFScalarTorch::DVCSCFFScalarTorch(const std::string& className)
        : DVCSConvolCoeffFunctionModule(className), m_pScalarCFF(0) {

    // The wrapped model owns any GPD dependence; this one adds none.
    setIsGPDModuleDependent(false);

    for (int k = 0; k < 4; ++k) {
        m_listOfCFFComputeFunctionAvailable.insert(
                std::make_pair(kTypes[k],
                        &PARTONS::DVCSConvolCoeffFunctionModule::computeCFF));
    }
}

DVCSCFFScalarTorch::DVCSCFFScalarTorch(const DVCSCFFScalarTorch& other)
        : DVCSConvolCoeffFunctionModule(other), m_pScalarCFF(other.m_pScalarCFF) {
}

DVCSCFFScalarTorch::~DVCSCFFScalarTorch() {
}

DVCSCFFScalarTorch* DVCSCFFScalarTorch::clone() const {
    return new DVCSCFFScalarTorch(*this);
}

void DVCSCFFScalarTorch::setScalarModule(
        PARTONS::DVCSConvolCoeffFunctionModule* pScalarCFF) {
    m_pScalarCFF = pScalarCFF;
}

// ---------------------------------------------------------------------------
// Scalar contract: behave like the model we wrap
// ---------------------------------------------------------------------------

std::complex<double> DVCSCFFScalarTorch::computeCFF() {

    if (!m_pScalarCFF)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "No scalar CFF module set. Call setScalarModule() first.");

    PARTONS::DVCSConvolCoeffFunctionKinematic ccfKin(m_xi, m_t, m_Q2, m_MuF2,
            m_MuR2);

    PARTONS::List<PARTONS::GPDType> gpdTypes;
    gpdTypes.add(PARTONS::GPDType(m_currentGPDComputeType));

    PARTONS::DVCSConvolCoeffFunctionResult result = m_pScalarCFF->compute(
            ccfKin, gpdTypes);

    const std::map<PARTONS::GPDType::Type, std::complex<double> >& values =
            result.getResultsByGpdType();
    std::map<PARTONS::GPDType::Type, std::complex<double> >::const_iterator it =
            values.find(m_currentGPDComputeType);

    return (it != values.end()) ? it->second : std::complex<double>(0., 0.);
}

// ---------------------------------------------------------------------------
// Tensor contract: the wrapped model, evaluated over the batch
// ---------------------------------------------------------------------------

DVCSCFFScalarTorch::AllCFFsTensorBatch
DVCSCFFScalarTorch::computeAllCFFsTensorBatch(const torch::Tensor& xi,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& muF2, const torch::Tensor& muR2) {

    if (!m_pScalarCFF)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "No scalar CFF module set. Call setScalarModule() first.");

    const int64_t N = xi.size(0);

    // [4][N] real and imaginary parts, in kTypes order.
    std::vector<std::vector<double> > re(4, std::vector<double>(N, 0.));
    std::vector<std::vector<double> > im(4, std::vector<double>(N, 0.));

    PARTONS::List<PARTONS::GPDType> gpdTypes;
    for (int k = 0; k < 4; ++k)
        gpdTypes.add(PARTONS::GPDType(kTypes[k]));

    for (int64_t i = 0; i < N; ++i) {

        // The kinematics arrive already converted -- the process module ran the
        // xi-converter and the scales module, as it does on the scalar path --
        // so there is nothing to do here but build the bean.
        PARTONS::DVCSConvolCoeffFunctionKinematic ccfKin(xi[i].item<double>(),
                t[i].item<double>(), Q2[i].item<double>(),
                muF2[i].item<double>(), muR2[i].item<double>());

        PARTONS::DVCSConvolCoeffFunctionResult result = m_pScalarCFF->compute(
                ccfKin, gpdTypes);

        // Read through the map rather than getResult(): a model that does not
        // provide a given CFF leaves it at zero instead of throwing, matching
        // what the network does for types outside its output layer.
        const std::map<PARTONS::GPDType::Type, std::complex<double> >& values =
                result.getResultsByGpdType();

        for (int k = 0; k < 4; ++k) {
            std::map<PARTONS::GPDType::Type, std::complex<double> >::const_iterator it =
                    values.find(kTypes[k]);
            if (it != values.end()) {
                re[k][i] = it->second.real();
                im[k][i] = it->second.imag();
            }
        }
    }

    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor cff[4];
    for (int k = 0; k < 4; ++k) {
        cff[k] = torch::complex(torch::tensor(re[k], f64),
                torch::tensor(im[k], f64)); // [N] complex double, no grad
    }

    AllCFFsTensorBatch cffs;
    cffs.H  = cff[0];
    cffs.E  = cff[1];
    cffs.Ht = cff[2];
    cffs.Et = cff[3];
    return cffs;
}