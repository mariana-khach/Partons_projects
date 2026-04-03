#include "../../../include/modules/fits_model/Partons0117NNFitsModel.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/channel/ChannelType.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/process/ProcessModule.h>
#include <partons/Partons.h>
#include <partons/ServiceObjectRegistry.h>
#include <iterator>
#include <map>
#include <utility>

#include "../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"
#include "../../../include/algorithms/neural_network/neural_network/NeuralNetwork.h"
#include "../../../include/modules/convol_coeff_function/DVCSCFFNN.h"
#include "../../../include/modules/minimizer_model/FitsMinimizerModule.h"
#include "../../../include/services/FitsService.h"

using namespace PARTONS;

const unsigned int Partons0117NNFitsModel::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new Partons0117NNFitsModel("Partons0117NNFitsModel"));

Partons0117NNFitsModel::Partons0117NNFitsModel(const std::string& className) :
        FitsModelModule(className), m_pDVCSNNCFFModel(0) {

    m_perceptonMap.clear();
    m_neuronMap.clear();
}

Partons0117NNFitsModel::Partons0117NNFitsModel(
        const Partons0117NNFitsModel &other) :
        FitsModelModule(other), m_pDVCSNNCFFModel(other.m_pDVCSNNCFFModel) {

    m_perceptonMap = other.m_perceptonMap;
    m_neuronMap = other.m_neuronMap;
}

Partons0117NNFitsModel::~Partons0117NNFitsModel() {
}

void Partons0117NNFitsModel::resolveObjectDependencies() {
    FitsModelModule::resolveObjectDependencies();
}

void Partons0117NNFitsModel::configure(
        const ElemUtils::Parameters &parameters) {
    FitsModelModule::configure(parameters);
}

Partons0117NNFitsModel* Partons0117NNFitsModel::clone() const {
    return new Partons0117NNFitsModel(*this);
}

void Partons0117NNFitsModel::initModule() {

    //run mother class
    FitsModelModule::initModule();

    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //get cff model
    if (pFitsService->getCFFModule(ChannelType::DVCS)->getClassName()
            != "DVCSCFFNN") {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "This fit model requires DVCSCFFNN to be used as DVCS/CFF model");
    }

    m_pDVCSNNCFFModel = static_cast<DVCSCFFNN*>(pFitsService->getCFFModule(
            ChannelType::DVCS));

    //register perceptrons and neurons
    std::vector<NeuralNetworkCell*>::const_iterator cell;
    std::vector<NeuralNetworkNeuron*>::const_iterator neuron;

    std::map<GPDType::Type, NeuralNetwork*>::const_iterator it;

    for (it = m_pDVCSNNCFFModel->getNeuralNetworks().begin();
            it != m_pDVCSNNCFFModel->getNeuralNetworks().end(); it++) {

        //get pointer to nn
        NeuralNetwork* nn = it->second;

        //get perceptrons
        for (cell = (nn->getNeuralNetworkCells()).begin();
                cell != (nn->getNeuralNetworkCells()).end(); cell++) {
            if ((*cell)->getType() == NeuralNetworkCellType::Perceptron)
                m_perceptonMap.push_back(static_cast<Perceptron*>(*cell));
        }

        //get neurons
        for (neuron = (nn->getNeuralNetworkNeurons()).begin();
                neuron != (nn->getNeuralNetworkNeurons()).end(); neuron++) {
            if ((*neuron)->getNeuralNetworkCellOut()->getType()
                            == NeuralNetworkCellType::Perceptron)
                m_neuronMap.push_back(
                        static_cast<NeuralNetworkNeuron*>(*neuron));
        }

        info(__func__,
                ElemUtils::Formatter()
                        << "Number of perceptrons after registration of CFF GPD type "
                        << GPDType(it->first).toString() << ": "
                        << m_perceptonMap.size());
        info(__func__,
                ElemUtils::Formatter()
                        << "Number of neurons after registration of CFF GPD type "
                        << GPDType(it->first).toString() << ": "
                        << m_neuronMap.size());
    }

    //set parameters
    for (itPerceptron = m_perceptonMap.begin();
            itPerceptron != m_perceptonMap.end(); itPerceptron++) {
        m_parameters.add(
                ElemUtils::Formatter() << "perceptron_"
                        << (itPerceptron - m_perceptonMap.begin()), 0.);
    }

    for (itNeuron = m_neuronMap.begin(); itNeuron != m_neuronMap.end();
            itNeuron++) {
        m_parameters.add(
                ElemUtils::Formatter() << "neuron_"
                        << (itNeuron - m_neuronMap.begin()), 0.);
    }
}

void Partons0117NNFitsModel::isModuleWellConfigured() {
    FitsModelModule::isModuleWellConfigured();
}

void Partons0117NNFitsModel::initParameters() {

    //TODO probably one should add FitsServiceObjectRegistry to PartonsFits to avoid casting
    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //get minimizer
    FitsMinimizerModule* pMinimizer = pFitsService->getMinimizer();

    //set parameters
    double initialStep = 1.E-1;

    for (itPerceptron = m_perceptonMap.begin();
            itPerceptron != m_perceptonMap.end(); itPerceptron++) {
        pMinimizer->SetLimitedVariable(
                ElemUtils::Formatter() << "perceptron_"
                        << (itPerceptron - m_perceptonMap.begin()), 0.,
                initialStep, std::pair<double, double>(-2., 2.));
    }

    for (itNeuron = m_neuronMap.begin(); itNeuron != m_neuronMap.end();
            itNeuron++) {
        pMinimizer->SetLimitedVariable(
                ElemUtils::Formatter() << "neuron_"
                        << (itNeuron - m_neuronMap.begin()), 0., initialStep,
                std::pair<double, double>(-2., 2.));
    }
}

void Partons0117NNFitsModel::updateParameters(const double* Var) {

    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //reset kinematics
    // resetPreviousKinematics() removed in new PARTONS

    //set parameters
    size_t parameter = 0;

    for (itPerceptron = m_perceptonMap.begin();
            itPerceptron != m_perceptonMap.end(); itPerceptron++) {
        (*itPerceptron)->setBias(Var[parameter]);
        parameter++;
    }

    for (itNeuron = m_neuronMap.begin(); itNeuron != m_neuronMap.end();
            itNeuron++) {

        (*itNeuron)->setWeight(Var[parameter]);
        parameter++;
    }
}
