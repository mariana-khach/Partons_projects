//
// Created by Mariana Khachatryan on 3/25/26.
//

#include "../../include/NNFit/DVCSCFFNNPytorch.h"

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
// computeCFF
// ---------------------------------------------------------------------------

std::complex<double> DVCSCFFNNPytorch::computeCFF() {

    if (!m_net)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Pytorch model has not been set. Call setModel() first.");

    // Determine output indices for this GPD type
    std::string gpdName = gpdTypeToName(m_currentGPDComputeType);
    std::string reName  = "Re" + gpdName;
    std::string imName  = "Im" + gpdName;

    int reIdx = -1, imIdx = -1;
    for (int k = 0; k < static_cast<int>(m_outputLayer.size()); ++k) {
        if (m_outputLayer[k] == reName) reIdx = k;
        if (m_outputLayer[k] == imName) imIdx = k;
    }

    // Build 1x3 input tensor: [xB, t, Q2]
    // xB derived from PARTONS skewness: xB = 2*xi / (1 + xi)
    double xB = 2.0 * m_xi / (1.0 + m_xi);

    torch::Tensor input = torch::zeros({1, 3});
    input[0][0] = static_cast<float>(xB);
    input[0][1] = static_cast<float>(m_t);
    input[0][2] = static_cast<float>(m_Q2);

    // Run inference
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        m_net->eval();
        output = m_net->forward(input);
    }

    double re = (reIdx >= 0) ? static_cast<double>(output[0][reIdx].item<float>()) : 0.0;
    double im = (imIdx >= 0) ? static_cast<double>(output[0][imIdx].item<float>()) : 0.0;

    return std::complex<double>(re, im);
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