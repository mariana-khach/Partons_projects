//
// Created by Mariana Khachatryan on 3/25/26.
//

#include "NNFit/theory/Modules/CFFs/DVCS/DVCSCFFNNPytorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/convol_coeff_function/ConvolCoeffFunctionModule.h>

#include <stdexcept>

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const unsigned int DVCSCFFNNPytorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCFFNNPytorch("DVCSCFFNNPytorch"));

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

DVCSCFFNNPytorch::DVCSCFFNNPytorch(const std::string& className)
        : DVCSConvolCoeffFunctionModule(className) {

    // This module computes CFFs directly — no GPD module needed
    setIsGPDModuleDependent(false);

    // Register all four CFF types, all routing to computeCFF()
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(PARTONS::GPDType::H,
                    &PARTONS::DVCSConvolCoeffFunctionModule::computeCFF));
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(PARTONS::GPDType::E,
                    &PARTONS::DVCSConvolCoeffFunctionModule::computeCFF));
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(PARTONS::GPDType::Ht,
                    &PARTONS::DVCSConvolCoeffFunctionModule::computeCFF));
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(PARTONS::GPDType::Et,
                    &PARTONS::DVCSConvolCoeffFunctionModule::computeCFF));
}

DVCSCFFNNPytorch::DVCSCFFNNPytorch(const DVCSCFFNNPytorch& other)
        : DVCSConvolCoeffFunctionModule(other),
          m_net(other.m_net),
          m_outputLayer(other.m_outputLayer) {
}

DVCSCFFNNPytorch::~DVCSCFFNNPytorch() {
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

DVCSCFFNNPytorch* DVCSCFFNNPytorch::clone() const {
    return new DVCSCFFNNPytorch(*this);
}

// ---------------------------------------------------------------------------
// PARTONS boilerplate
// ---------------------------------------------------------------------------

void DVCSCFFNNPytorch::configure(const ElemUtils::Parameters& parameters) {
    DVCSConvolCoeffFunctionModule::configure(parameters);
}

void DVCSCFFNNPytorch::resolveObjectDependencies() {
    DVCSConvolCoeffFunctionModule::resolveObjectDependencies();
}

void DVCSCFFNNPytorch::prepareSubModules(
        const std::map<std::string, PARTONS::BaseObjectData>& subModulesData) {
    DVCSConvolCoeffFunctionModule::prepareSubModules(subModulesData);
}

void DVCSCFFNNPytorch::initModule() {
    DVCSConvolCoeffFunctionModule::initModule();
}

void DVCSCFFNNPytorch::isModuleWellConfigured() {
    DVCSConvolCoeffFunctionModule::isModuleWellConfigured();
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void DVCSCFFNNPytorch::setModel(CFFNNModel net,
        const std::vector<std::string>& outputLayer) {
    m_net         = net;
    m_outputLayer = outputLayer;
}

// ---------------------------------------------------------------------------
// computeAllCFFsTensor — one NN forward, returns all 8 CFF components.
// Used by the tensor pipeline to avoid the 4× redundant per-GPD-type
// forward passes that computeCFFTensor(type) would otherwise incur.
// ---------------------------------------------------------------------------

DVCSCFFNNPytorch::AllCFFsTensor DVCSCFFNNPytorch::computeAllCFFsTensor() {

    if (!m_net)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Pytorch model has not been set. Call setModel() first.");

    // xB derived from PARTONS skewness: xB = 2*xi / (1 + xi)
    const double xB = 2.0 * m_xi / (1.0 + m_xi);

    torch::Tensor input = torch::zeros({1, 3});
    input[0][0] = static_cast<float>(xB);
    input[0][1] = static_cast<float>(m_t);
    input[0][2] = static_cast<float>(m_Q2);

    const torch::Tensor output = m_net->forward(input);

    auto select = [&](const std::string& name) -> torch::Tensor {
        for (int k = 0; k < static_cast<int>(m_outputLayer.size()); ++k)
            if (m_outputLayer[k] == name)
                return output[0][k];
        return torch::zeros({});
    };

    AllCFFsTensor cffs;
    cffs.H_re  = select("ReH");  cffs.H_im  = select("ImH");
    cffs.E_re  = select("ReE");  cffs.E_im  = select("ImE");
    cffs.Ht_re = select("ReHt"); cffs.Ht_im = select("ImHt");
    cffs.Et_re = select("ReEt"); cffs.Et_im = select("ImEt");
    return cffs;
}

void DVCSCFFNNPytorch::setupKinematics(double xi, double t, double Q2) {
    // Direct write to the inherited protected members. Equivalent to what
    // PARTONS' scalar pipeline does inside computeConvolCoeffFunction()
    // before dispatching to computeCFF() — but without the NN forward.
    m_xi = xi;
    m_t  = t;
    m_Q2 = Q2;
}

// ---------------------------------------------------------------------------
// computeCFFTensor — single source of truth: builds the autograd-connected
// Re/Im tensors for the GPD type currently requested by PARTONS.
// ---------------------------------------------------------------------------

std::pair<torch::Tensor, torch::Tensor> DVCSCFFNNPytorch::computeCFFTensor(
        PARTONS::GPDType::Type gpdType) {

    if (!m_net)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Pytorch model has not been set. Call setModel() first.");

    std::string gpdName = gpdTypeToName(gpdType);
    std::string reName  = "Re" + gpdName;
    std::string imName  = "Im" + gpdName;

    int reIdx = -1, imIdx = -1;
    for (int k = 0; k < static_cast<int>(m_outputLayer.size()); ++k) {
        if (m_outputLayer[k] == reName) reIdx = k;
        if (m_outputLayer[k] == imName) imIdx = k;
    }

    // xB derived from PARTONS skewness: xB = 2*xi / (1 + xi)
    double xB = 2.0 * m_xi / (1.0 + m_xi);

    torch::Tensor input = torch::zeros({1, 3});
    input[0][0] = static_cast<float>(xB);
    input[0][1] = static_cast<float>(m_t);
    input[0][2] = static_cast<float>(m_Q2);

    torch::Tensor output = m_net->forward(input);

    torch::Tensor re = (reIdx >= 0) ? output[0][reIdx] : torch::zeros({});
    torch::Tensor im = (imIdx >= 0) ? output[0][imIdx] : torch::zeros({});

    return {re, im};
}

// ---------------------------------------------------------------------------
// computeCFF — PARTONS entry point. Wraps computeCFFTensor() under
// NoGradGuard so the autograd graph isn't built for the inference path.
// ---------------------------------------------------------------------------

std::complex<double> DVCSCFFNNPytorch::computeCFF() {

    torch::NoGradGuard no_grad;
    m_net->eval();

    auto [re, im] = computeCFFTensor(m_currentGPDComputeType);
    return std::complex<double>(
            static_cast<double>(re.item<float>()),
            static_cast<double>(im.item<float>()));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string DVCSCFFNNPytorch::gpdTypeToName(PARTONS::GPDType::Type type) const {
    switch (type) {
        case PARTONS::GPDType::H:  return "H";
        case PARTONS::GPDType::E:  return "E";
        case PARTONS::GPDType::Ht: return "Ht";
        case PARTONS::GPDType::Et: return "Et";
        default:
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unknown GPD type: " << type);
    }
}