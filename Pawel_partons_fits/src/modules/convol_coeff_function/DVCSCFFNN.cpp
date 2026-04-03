#include "../../../include/modules/convol_coeff_function/DVCSCFFNN.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectRegistry.h>
#include <cmath>
#include <vector>

#include "../../../include/algorithms/neural_network/beans/ActivationFunctionType.h"
#include "../../../include/algorithms/neural_network/beans/CombinationFunctionType.h"
#include "../../../include/algorithms/neural_network/beans/Data.h"
#include "../../../include/algorithms/neural_network/beans/InitializationType.h"
#include "../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"
#include "../../../include/algorithms/neural_network/beans/ScalingFunctionType.h"
#include "../../../include/algorithms/neural_network/neural_network_cell/ScalingCell.h"
#include "../../../include/algorithms/neural_network/neural_network_layer/NeuralNetworkLayer.h"

using namespace PARTONS;

const unsigned int DVCSCFFNN::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCFFNN("DVCSCFFNN"));

DVCSCFFNN::DVCSCFFNN(const std::string &className) :
        DVCSConvolCoeffFunctionModule(className) {

    //mark as GPDModule independent
    setIsGPDModuleDependent(false);

    //availible CFFs
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::H,
                    &DVCSConvolCoeffFunctionModule::computeCFF));

    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::E,
                    &DVCSConvolCoeffFunctionModule::computeCFF));

    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::Ht,
                    &DVCSConvolCoeffFunctionModule::computeCFF));

    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::Et,
                    &DVCSConvolCoeffFunctionModule::computeCFF));

    NeuralNetwork* nullNN = 0;

    m_neuralNetworks.insert(std::make_pair(GPDType::H, nullNN));
    m_neuralNetworks.insert(std::make_pair(GPDType::E, nullNN));
    m_neuralNetworks.insert(std::make_pair(GPDType::Ht, nullNN));
    m_neuralNetworks.insert(std::make_pair(GPDType::Et, nullNN));

    m_rangeLog10Xi = std::make_pair(-2., 0.);
    m_rangeT = std::make_pair(-0.5, 0.);
    m_rangeQ2 = std::make_pair(1., 5.);
    m_rangeReCFF.insert(std::make_pair(GPDType::H, std::make_pair(-50., 20.)));
    m_rangeReCFF.insert(std::make_pair(GPDType::E, std::make_pair(-20., 20.)));
    m_rangeReCFF.insert(std::make_pair(GPDType::Ht, std::make_pair(-10., 15.)));
    m_rangeReCFF.insert(
            std::make_pair(GPDType::Et, std::make_pair(-500., 500.)));
    m_rangeXiImCFF.insert(std::make_pair(GPDType::H, std::make_pair(-1., 3.)));
    m_rangeXiImCFF.insert(
            std::make_pair(GPDType::E, std::make_pair(-10., 10.)));
    m_rangeXiImCFF.insert(
            std::make_pair(GPDType::Ht, std::make_pair(-0.5, 1.)));
    m_rangeXiImCFF.insert(
            std::make_pair(GPDType::Et, std::make_pair(-10., 10.)));
}

DVCSCFFNN::DVCSCFFNN(const DVCSCFFNN &other) :
        DVCSConvolCoeffFunctionModule(other) {

    NeuralNetwork* nullNN = 0;

    std::map<GPDType::Type, NeuralNetwork*>::const_iterator it;

    for (it = other.m_neuralNetworks.begin();
            it != other.m_neuralNetworks.end(); it++) {
        if (it->second) {
            m_neuralNetworks.insert(
                    std::make_pair(it->first, it->second->clone()));
        } else {
            m_neuralNetworks.insert(std::make_pair(it->first, nullNN));
        }
    }

    m_rangeLog10Xi = other.m_rangeLog10Xi;
    m_rangeT = other.m_rangeT;
    m_rangeQ2 = other.m_rangeQ2;
    m_rangeReCFF = other.m_rangeReCFF;
    m_rangeXiImCFF = other.m_rangeXiImCFF;
}

DVCSCFFNN* DVCSCFFNN::clone() const {
    return new DVCSCFFNN(*this);
}

DVCSCFFNN::~DVCSCFFNN() {

    std::map<GPDType::Type, NeuralNetwork*>::iterator it;

    for (it = m_neuralNetworks.begin(); it != m_neuralNetworks.end(); it++) {
        if (it->second) {
            delete it->second;
            it->second = 0;
        }
    }
}

void DVCSCFFNN::configure(const ElemUtils::Parameters &parameters) {
    DVCSConvolCoeffFunctionModule::configure(parameters);
}

void DVCSCFFNN::resolveObjectDependencies() {

    DVCSConvolCoeffFunctionModule::resolveObjectDependencies();

    //definition of neural nets
    std::vector<unsigned int> perceptronLayers;

    //H
    perceptronLayers.clear();
    perceptronLayers.push_back(5);
    perceptronLayers.push_back(3);

    m_neuralNetworks.find(GPDType::H)->second = new NeuralNetwork(3, 2,
            perceptronLayers, ActivationFunctionType::Hyperbolic,
            CombinationFunctionType::ScalarProduct,
            ActivationFunctionType::Linear,
            CombinationFunctionType::ScalarProduct, ScalingFunctionType::MinMax,
            InitializationType::Random);

    //E
    perceptronLayers.clear();
    perceptronLayers.push_back(5);
    perceptronLayers.push_back(3);

    m_neuralNetworks.find(GPDType::E)->second = new NeuralNetwork(3, 2,
            perceptronLayers, ActivationFunctionType::Hyperbolic,
            CombinationFunctionType::ScalarProduct,
            ActivationFunctionType::Linear,
            CombinationFunctionType::ScalarProduct, ScalingFunctionType::MinMax,
            InitializationType::Random);

    //Ht
    perceptronLayers.clear();
    perceptronLayers.push_back(5);
    perceptronLayers.push_back(3);

    m_neuralNetworks.find(GPDType::Ht)->second = new NeuralNetwork(3, 2,
            perceptronLayers, ActivationFunctionType::Hyperbolic,
            CombinationFunctionType::ScalarProduct,
            ActivationFunctionType::Linear,
            CombinationFunctionType::ScalarProduct, ScalingFunctionType::MinMax,
            InitializationType::Random);

    //Et
    perceptronLayers.clear();
    perceptronLayers.push_back(5);
    perceptronLayers.push_back(3);

    m_neuralNetworks.find(GPDType::Et)->second = new NeuralNetwork(3, 2,
            perceptronLayers, ActivationFunctionType::Hyperbolic,
            CombinationFunctionType::ScalarProduct,
            ActivationFunctionType::Linear,
            CombinationFunctionType::ScalarProduct, ScalingFunctionType::MinMax,
            InitializationType::Random);

    //scaling
    NeuralNetworkLayer* pNNLayer;
    ScalingCell* pNNCell;

    std::map<GPDType::Type, NeuralNetwork*>::iterator it;

    for (it = m_neuralNetworks.begin(); it != m_neuralNetworks.end(); it++) {

        //input
        pNNLayer = it->second->getNeuralNetworkLayers().at(1);

        //xi
        if (pNNLayer->getCells().at(0)->getType()
                != NeuralNetworkCellType::ScalingCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No scaling cell type");
        }
        pNNCell = static_cast<ScalingCell*>(pNNLayer->getCells().at(0));
        pNNCell->setScalingParameters(m_rangeLog10Xi);

        //t
        if (pNNLayer->getCells().at(1)->getType()
                != NeuralNetworkCellType::ScalingCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No scaling cell type");
        }
        pNNCell = static_cast<ScalingCell*>(pNNLayer->getCells().at(1));
        pNNCell->setScalingParameters(m_rangeT);

        //Q2
        if (pNNLayer->getCells().at(2)->getType()
                != NeuralNetworkCellType::ScalingCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No scaling cell type");
        }
        pNNCell = static_cast<ScalingCell*>(pNNLayer->getCells().at(2));
        pNNCell->setScalingParameters(m_rangeQ2);

        //output
        pNNLayer = it->second->getNeuralNetworkLayers().at(
                it->second->getNeuralNetworkLayers().size() - 2);

        //Re
        if (pNNLayer->getCells().at(0)->getType()
                != NeuralNetworkCellType::ScalingCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No scaling cell type");
        }
        pNNCell = static_cast<ScalingCell*>(pNNLayer->getCells().at(0));
        pNNCell->setScalingParameters(m_rangeReCFF.find(it->first)->second);

        //Im
        if (pNNLayer->getCells().at(1)->getType()
                != NeuralNetworkCellType::ScalingCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No scaling cell type");
        }
        pNNCell = static_cast<ScalingCell*>(pNNLayer->getCells().at(1));
        pNNCell->setScalingParameters(m_rangeXiImCFF.find(it->first)->second);
    }

    info(__func__, (m_neuralNetworks.find(GPDType::H)->second)->toString());
}

void DVCSCFFNN::prepareSubModules(
        const std::map<std::string, BaseObjectData>& subModulesData) {
    DVCSConvolCoeffFunctionModule::prepareSubModules(subModulesData);
}

void DVCSCFFNN::initModule() {
    DVCSConvolCoeffFunctionModule::initModule();
}

void DVCSCFFNN::isModuleWellConfigured() {
    DVCSConvolCoeffFunctionModule::isModuleWellConfigured();
}

std::complex<double> DVCSCFFNN::computeCFF() {

    std::map<GPDType::Type, NeuralNetwork*>::iterator it =
            m_neuralNetworks.find(m_currentGPDComputeType);

    if (it == m_neuralNetworks.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Calculation for GPD type "
                        << GPDType(m_currentGPDComputeType).toString()
                        << " unsupported");
    }

    if (!it->second) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Null pointer to NN representing CFF for GPD "
                        << GPDType(m_currentGPDComputeType).toString());
    }

    Data dataInput(3);

    std::vector<double> dataSingle;
    dataSingle.push_back(log10(m_xi));
    dataSingle.push_back(m_t);
    dataSingle.push_back(m_Q2);
    dataInput.addData(dataSingle);

    Data dataOutput = (it->second)->evaluate(dataInput);

    double r_part = ((dataOutput.getData().find(0))->second).at(0);
    double i_part = ((dataOutput.getData().find(1))->second).at(0) / m_xi;

    return std::complex<double>(r_part, i_part);
}

const std::map<GPDType::Type, NeuralNetwork*>& DVCSCFFNN::getNeuralNetworks() const {
    return m_neuralNetworks;
}
