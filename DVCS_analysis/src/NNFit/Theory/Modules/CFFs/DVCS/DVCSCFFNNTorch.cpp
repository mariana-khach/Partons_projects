//
// Created by Mariana Khachatryan on 3/25/26.
//

#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/convol_coeff_function/ConvolCoeffFunctionModule.h>

#include <cmath>
#include <iostream>
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
// Batched NN forward (single source of truth for the CFF value itself, for both
// the scalar and tensor paths; computeCFFTensor() is an N=1 wrapper).
// ---------------------------------------------------------------------------

torch::Tensor DVCSCFFNNTorch::forwardNNBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2) {

    if (!m_net)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Pytorch model has not been set. Call setModel() first.");

    // Stack [xB,t,Q2] as an [N,3] input.
    torch::Tensor input = torch::stack({xB, t, Q2}, /*dim=*/1).to(torch::kFloat32);

    // Apply the same per-feature min-max scaling fitted on the training set.
    // Matches (x - xMin) / (xMax - xMin) from CFF_NN_Fitter::train_nn().
    // Skipped if no scaling was injected (raw features).
    if (m_xMin.defined() && m_xMax.defined()) {
        torch::Tensor denom = (m_xMax - m_xMin).clamp_min(1e-8f);
        input = (input - m_xMin) / denom;
    }

    // Train/eval mode is the CALLER's to set (EvalModeGuard for an inference
    // call, net->train() for a training step) -- this forward serves both. It
    // used to force eval() here, which made training mode unreachable: harmless
    // for a Linear/Tanh net, silently wrong the day a Dropout/BatchNorm layer
    // is added.
    //
    // Network runs in float32; promote to float64 so downstream BMJ12
    // arithmetic matches the scalar (double) pipeline.
    torch::Tensor output = m_net->forward(input).to(torch::kFloat64); // [N, Nout]

    // CFF = xB^m_xPow * NNet_output, broadcast over the N axis.
    torch::Tensor xPowFactor = torch::pow(xB.to(torch::kFloat64), m_xPow); // [N]
    return output * xPowFactor.unsqueeze(1); // [N, Nout]
}

torch::Tensor DVCSCFFNNTorch::cffComponentTensorBatch(const torch::Tensor& output,
        const std::string& name) const {

    const std::string reName = "Re" + name;
    const std::string imName = "Im" + name;

    int reIdx = -1, imIdx = -1;
    for (int k = 0; k < static_cast<int>(m_outputLayer.size()); ++k) {
        if (m_outputLayer[k] == reName) reIdx = k;
        if (m_outputLayer[k] == imName) imIdx = k;
    }

    const int64_t N = output.size(0);
    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor re = (reIdx >= 0) ? output.select(1, reIdx) : torch::zeros({N}, f64);
    torch::Tensor im = (imIdx >= 0) ? output.select(1, imIdx) : torch::zeros({N}, f64);

    return torch::complex(re, im); // [N] complex double, grad-tracked
}

// ---------------------------------------------------------------------------
// Tensor CFFs
// ---------------------------------------------------------------------------

DVCSCFFNNTorch::AllCFFsTensorBatch DVCSCFFNNTorch::computeAllCFFsTensorBatch(
        const torch::Tensor& xi, const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& /* muF2 */, const torch::Tensor& /* muR2 */) {

    // Back to the network's own feature. The scalar chain hands every CFF
    // module the CCF kinematics (xi, t, Q2, muF2, muR2), so the tensor chain
    // does too; a module parameterized in xB converts here. The scales are
    // ignored -- the network is scale-blind by construction.
    torch::Tensor xB = 2. * xi / (1. + xi);
    torch::Tensor output = forwardNNBatch(xB, t, Q2);
    AllCFFsTensorBatch cffs;
    cffs.H  = cffComponentTensorBatch(output, "H");
    cffs.E  = cffComponentTensorBatch(output, "E");
    cffs.Ht = cffComponentTensorBatch(output, "Ht");
    cffs.Et = cffComponentTensorBatch(output, "Et");
    return cffs;
}

torch::Tensor DVCSCFFNNTorch::computeCFFTensorBatch(PARTONS::GPDType::Type type,
        const torch::Tensor& xB, const torch::Tensor& t, const torch::Tensor& Q2) {
    std::string name;
    switch (type) {
    case PARTONS::GPDType::H:  name = "H";  break;
    case PARTONS::GPDType::E:  name = "E";  break;
    case PARTONS::GPDType::Ht: name = "Ht"; break;
    case PARTONS::GPDType::Et: name = "Et"; break;
    default:
        // Types the NN does not parametrize (e.g. transversity / twist-3)
        return torch::complex(torch::zeros({xB.size(0)}, torch::kFloat64),
                torch::zeros({xB.size(0)}, torch::kFloat64));
    }
    return cffComponentTensorBatch(forwardNNBatch(xB, t, Q2), name);
}

// ---------------------------------------------------------------------------
// Single-point CFF -- N=1 wrapper around the batched implementation above.
// Kept because PARTONS' base-scalar pipeline reaches it through computeCFF()
// (one GPD type at a time), e.g. when observ_calc() drives the base
// DVCSProcessBMJ12. The tensor chain uses the batched methods directly.
// ---------------------------------------------------------------------------

torch::Tensor DVCSCFFNNTorch::computeCFFTensor(PARTONS::GPDType::Type type) {
    double xB = 2.0 * m_xi / (1.0 + m_xi);
    torch::Tensor xBT = torch::full({1}, xB, torch::kFloat64);
    torch::Tensor tT  = torch::full({1}, m_t, torch::kFloat64);
    torch::Tensor Q2T = torch::full({1}, m_Q2, torch::kFloat64);

    return computeCFFTensorBatch(type, xBT, tT, Q2T)[0];
}

// ---------------------------------------------------------------------------
// computeCFF — scalar wrapper over the tensor path (no gradient)
//
// This is PARTONS' scalar entry point, so it establishes both inference
// conditions itself: no autograd graph (NoGradGuard) and eval mode
// (EvalModeGuard, restored on exit so a scalar call mid-training cannot leave
// the shared net in the wrong mode).
// ---------------------------------------------------------------------------

std::complex<double> DVCSCFFNNTorch::computeCFF() {
    torch::NoGradGuard no_grad;
    EvalModeGuard eval_mode(m_net);
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