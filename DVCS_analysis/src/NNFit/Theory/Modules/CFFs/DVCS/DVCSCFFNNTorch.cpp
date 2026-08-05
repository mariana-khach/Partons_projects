//
// Created by Mariana Khachatryan on 3/25/26.
//

#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/convol_coeff_function/ConvolCoeffFunctionModule.h>

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const unsigned int DVCSCFFNNTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCFFNNTorch("DVCSCFFNNTorch"));

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

DVCSCFFNNTorch::DVCSCFFNNTorch(const std::string& className)
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

DVCSCFFNNTorch::DVCSCFFNNTorch(const DVCSCFFNNTorch& other)
        : DVCSConvolCoeffFunctionModule(other),
          m_net(other.m_net),
          m_outputLayer(other.m_outputLayer),
          m_xMin(other.m_xMin),
          m_xMax(other.m_xMax),
          m_xPow(other.m_xPow) {
}

DVCSCFFNNTorch::~DVCSCFFNNTorch() {
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

DVCSCFFNNTorch* DVCSCFFNNTorch::clone() const {
    return new DVCSCFFNNTorch(*this);
}

// ---------------------------------------------------------------------------
// PARTONS boilerplate
// ---------------------------------------------------------------------------

void DVCSCFFNNTorch::configure(const ElemUtils::Parameters& parameters) {
    DVCSConvolCoeffFunctionModule::configure(parameters);
}

void DVCSCFFNNTorch::resolveObjectDependencies() {
    DVCSConvolCoeffFunctionModule::resolveObjectDependencies();
}

void DVCSCFFNNTorch::prepareSubModules(
        const std::map<std::string, PARTONS::BaseObjectData>& subModulesData) {
    DVCSConvolCoeffFunctionModule::prepareSubModules(subModulesData);
}

void DVCSCFFNNTorch::initModule() {
    DVCSConvolCoeffFunctionModule::initModule();
}

void DVCSCFFNNTorch::isModuleWellConfigured() {
    DVCSConvolCoeffFunctionModule::isModuleWellConfigured();
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void DVCSCFFNNTorch::setModel(CFFNNModel net,
        const std::vector<std::string>& outputLayer,
        const torch::Tensor& xMin, const torch::Tensor& xMax, double xPow) {
    m_net         = net;
    m_outputLayer = outputLayer;
    m_xMin        = xMin;
    m_xMax        = xMax;
    m_xPow        = xPow;
}

// ---------------------------------------------------------------------------
// Kinematics (tensor path)
// ---------------------------------------------------------------------------

void DVCSCFFNNTorch::setupKinematicsTorch(double xi, double t, double Q2) {
    m_xi  = xi;
    m_t   = t;
    m_Q2  = Q2;
}

// ---------------------------------------------------------------------------
// NN forward (single source of truth for both scalar and tensor paths)
// ---------------------------------------------------------------------------

torch::Tensor DVCSCFFNNTorch::forwardNN() {

    if (!m_net)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Pytorch model has not been set. Call setModel() first.");

    // Build 1x3 input tensor: [xB, t, Q2]
    // xB derived from PARTONS skewness: xB = 2*xi / (1 + xi)
    double xB = 2.0 * m_xi / (1.0 + m_xi);

    torch::Tensor input = torch::zeros({1, 3});
    input[0][0] = static_cast<float>(xB);
    input[0][1] = static_cast<float>(m_t);
    input[0][2] = static_cast<float>(m_Q2);

    // Apply the same per-feature min-max scaling fitted on the training set.
    // Matches (x - xMin) / (xMax - xMin) from CFF_NN_Fitter::train_nn().
    // Skipped if no scaling was injected (raw features).
    if (m_xMin.defined() && m_xMax.defined()) {
        torch::Tensor denom = (m_xMax - m_xMin).clamp_min(1e-8f);
        input = (input - m_xMin) / denom;
    }

    m_net->eval();
    // Network runs in float32; promote to float64 so downstream BMJ12
    // arithmetic matches the scalar (double) pipeline.
    torch::Tensor output = m_net->forward(input).to(torch::kFloat64);

    // CFF = xB^m_xPow * NNet_output (m_xPow defaults to 0, i.e. no rescaling).
    return output * std::pow(xB, m_xPow);
}

torch::Tensor DVCSCFFNNTorch::cffComponentTensor(const torch::Tensor& output,
        const std::string& name) const {

    const std::string reName = "Re" + name;
    const std::string imName = "Im" + name;

    int reIdx = -1, imIdx = -1;
    for (int k = 0; k < static_cast<int>(m_outputLayer.size()); ++k) {
        if (m_outputLayer[k] == reName) reIdx = k;
        if (m_outputLayer[k] == imName) imIdx = k;
    }

    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor re = (reIdx >= 0) ? output[0][reIdx] : torch::zeros({}, f64);
    torch::Tensor im = (imIdx >= 0) ? output[0][imIdx] : torch::zeros({}, f64);

    return torch::complex(re, im); // 0-d complex double, grad-tracked
}

// ---------------------------------------------------------------------------
// Tensor CFFs
// ---------------------------------------------------------------------------

DVCSCFFNNTorch::AllCFFsTensor DVCSCFFNNTorch::computeAllCFFsTensor() {
    torch::Tensor output = forwardNN();
    AllCFFsTensor cffs;
    cffs.H  = cffComponentTensor(output, "H");
    cffs.E  = cffComponentTensor(output, "E");
    cffs.Ht = cffComponentTensor(output, "Ht");
    cffs.Et = cffComponentTensor(output, "Et");
    return cffs;
}

torch::Tensor DVCSCFFNNTorch::computeCFFTensor(PARTONS::GPDType::Type type) {
    std::string name;
    switch (type) {
    case PARTONS::GPDType::H:  name = "H";  break;
    case PARTONS::GPDType::E:  name = "E";  break;
    case PARTONS::GPDType::Ht: name = "Ht"; break;
    case PARTONS::GPDType::Et: name = "Et"; break;
    default:
        // Types the NN does not parametrize (e.g. transversity / twist-3)
        return torch::complex(torch::zeros({}, torch::kFloat64),
                torch::zeros({}, torch::kFloat64));
    }
    return cffComponentTensor(forwardNN(), name);
}

// ---------------------------------------------------------------------------
// computeCFF — scalar wrapper over the tensor path (no gradient)
// ---------------------------------------------------------------------------

std::complex<double> DVCSCFFNNTorch::computeCFF() {
    torch::NoGradGuard no_grad;
    torch::Tensor cff = computeCFFTensor(m_currentGPDComputeType);
    return std::complex<double>(torch::real(cff).item<double>(),
            torch::imag(cff).item<double>());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string DVCSCFFNNTorch::gpdTypeToName(PARTONS::GPDType::Type type) const {
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