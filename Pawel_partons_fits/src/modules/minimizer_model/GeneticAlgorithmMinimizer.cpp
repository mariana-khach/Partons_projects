#include "../../../include/modules/minimizer_model/GeneticAlgorithmMinimizer.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/GenericType.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectRegistry.h>
#include <vector>

#include "../../../include/algorithms/genetic_algorithm/GeneticAlgorithm.h"

using namespace PARTONS;

const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_POPULATION_SIZE_MIN =
        "ga_minimizer_population_size_min";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_POPULATION_SIZE_MAX =
        "ga_minimizer_population_size_max";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_POPULATION_SIZE_SLOPE =
        "ga_minimizer_population_size_slope";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_BEST_FRACTION =
        "ga_minimizer_best_fraction";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_CROSS_SLOPE =
        "ga_minimizer_cross_slope";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_PROB_MAX =
        "ga_minimizer_mutation_A_prob_max";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_B_PROB_MAX =
        "ga_minimizer_mutation_B_prob_max";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_SIZE =
        "ga_minimizer_mutation_A_size";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_RESISTANCE =
        "ga_minimizer_mutation_A_resistance";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_B_RESISTANCE =
        "ga_minimizer_mutation_B_resistance";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_SLOPE =
        "ga_minimizer_mutation_A_slope";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_B_SLOPE =
        "ga_minimizer_mutation_B_slope";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_STOP_CONDITION_A =
        "ga_minimizer_max_iterations";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_STOP_CONDITION_B1 =
        "ga_minimizer_min_precision_sampling";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_STOP_CONDITION_B2 =
        "ga_minimizer_min_precision";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_START_FROM_DUMP_FILE =
        "ga_minimizer_start_from_dump_file";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_BEST_TO_FILE_EVERY =
        "ga_minimizer_dump_best_to_file_every";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_BEST_TO_FILE_PATH =
        "ga_minimizer_dump_best_to_file_path";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_POPULATION_TO_FILE_EVERY =
        "ga_minimizer_dump_population_to_file_every";
const std::string GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_POPULATION_TO_FILE_PATH =
        "ga_minimizer_dump_population_to_file_path";

const size_t GeneticAlgorithmMinimizer::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new GeneticAlgorithmMinimizer("GeneticAlgorithmMinimizer"));

GeneticAlgorithmMinimizer::GeneticAlgorithmMinimizer(
        const std::string& className) :
        FitsMinimizerModule(className) {
    m_geneticAlgorithm = new GeneticAlgorithm();
}

GeneticAlgorithmMinimizer::GeneticAlgorithmMinimizer(
        const GeneticAlgorithmMinimizer& other) :
        FitsMinimizerModule(other) {
    m_geneticAlgorithm = new GeneticAlgorithm(*other.m_geneticAlgorithm);
}

GeneticAlgorithmMinimizer::~GeneticAlgorithmMinimizer() {

    if (m_geneticAlgorithm) {
        delete m_geneticAlgorithm;
        m_geneticAlgorithm = 0;
    }
}

void GeneticAlgorithmMinimizer::resolveObjectDependencies() {
    FitsMinimizerModule::resolveObjectDependencies();
}

void GeneticAlgorithmMinimizer::configure(
        const ElemUtils::Parameters& parameters) {

    FitsMinimizerModule::configure(parameters);

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_POPULATION_SIZE_MIN)) {

        m_geneticAlgorithm->setPopulationSizeMin(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toUInt());

        info(__func__,
                ElemUtils::Formatter() << "Minimum population size changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_POPULATION_SIZE_MAX)) {

        m_geneticAlgorithm->setPopulationSizeMax(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toUInt());

        info(__func__,
                ElemUtils::Formatter() << "Maximum population size changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_POPULATION_SIZE_SLOPE)) {

        m_geneticAlgorithm->setPopulationSizeSlope(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Population size slope changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_BEST_FRACTION)) {

        m_geneticAlgorithm->setBestFraction(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Best fraction changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_CROSS_SLOPE)) {

        m_geneticAlgorithm->setCrossSlope(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter()
                        << "Cross exponential factor changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_PROB_MAX)) {

        m_geneticAlgorithm->setMutationAProbMax(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter()
                        << "Maximum mutation A probability changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_B_PROB_MAX)) {

        m_geneticAlgorithm->setMutationBProbMax(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter()
                        << "Maximum mutation B probability changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_SIZE)) {

        m_geneticAlgorithm->setMutationASize(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Mutation A size changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_RESISTANCE)) {

        m_geneticAlgorithm->setMutationAResistance(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Mutation resistance A changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_B_RESISTANCE)) {

        m_geneticAlgorithm->setMutationBResistance(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Mutation resistance B changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_A_SLOPE)) {

        m_geneticAlgorithm->setMutationASlope(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Mutation A slope changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_MUTATION_B_SLOPE)) {

        m_geneticAlgorithm->setMutationBSlope(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Mutation B slope changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_STOP_CONDITION_A)) {

        m_geneticAlgorithm->setStopConditionA(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toUInt());

        info(__func__,
                ElemUtils::Formatter()
                        << "Maximum number of iterations changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_STOP_CONDITION_B1)) {

        m_geneticAlgorithm->setStopConditionB1(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toUInt());

        info(__func__,
                ElemUtils::Formatter()
                        << "Minimum precision sampling changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_STOP_CONDITION_B2)) {

        m_geneticAlgorithm->setStopConditionB2(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Minimum precision changed to: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_START_FROM_DUMP_FILE)) {

        m_geneticAlgorithm->readPopulationFromFile(
                parameters.getLastAvailable().getString());

        info(__func__,
                ElemUtils::Formatter() << "Population started from file: "
                        << parameters.getLastAvailable().getString());
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_BEST_TO_FILE_EVERY)) {

        m_geneticAlgorithm->setDumpBestToFileEvery(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toSize_t());

        info(__func__,
                ElemUtils::Formatter()
                        << "Drop best candidate to file every iteration: "
                        << parameters.getLastAvailable().toSize_t());

        if (parameters.isAvailable(
                GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_BEST_TO_FILE_PATH)) {

            m_geneticAlgorithm->setDumpBestToFilePath(
                    ElemUtils::GenericType(
                            parameters.getLastAvailable().getString()).getString());

            info(__func__,
                    ElemUtils::Formatter() << "Drop best candidate to file: "
                            << parameters.getLastAvailable().getString());
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <kineamtics type=\""
                            << PARAMETER_NAME_DUMP_BEST_TO_FILE_PATH
                            << "\"> element ; Or check case");
        }
    }

    if (parameters.isAvailable(
            GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_POPULATION_TO_FILE_EVERY)) {

        m_geneticAlgorithm->setDumpPopulationToFileEvery(
                ElemUtils::GenericType(
                        parameters.getLastAvailable().getString()).toSize_t());

        info(__func__,
                ElemUtils::Formatter()
                        << "Drop population to file every iteration: "
                        << parameters.getLastAvailable().toSize_t());

        if (parameters.isAvailable(
                GeneticAlgorithmMinimizer::PARAMETER_NAME_DUMP_POPULATION_TO_FILE_PATH)) {

            m_geneticAlgorithm->setDumpPopulationToFilePath(
                    ElemUtils::GenericType(
                            parameters.getLastAvailable().getString()).getString());

            info(__func__,
                    ElemUtils::Formatter() << "Drop population to file: "
                            << parameters.getLastAvailable().getString());
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <kineamtics type=\""
                            << PARAMETER_NAME_DUMP_POPULATION_TO_FILE_PATH
                            << "\"> element ; Or check case");
        }
    }

}

GeneticAlgorithmMinimizer* GeneticAlgorithmMinimizer::clone() const {
    return new GeneticAlgorithmMinimizer(*this);
}

//TODO not called anywhere
void GeneticAlgorithmMinimizer::initModule() {
    FitsMinimizerModule::initModule();
}

//TODO not called anywhere
void GeneticAlgorithmMinimizer::isModuleWellConfigured() {
    FitsMinimizerModule::isModuleWellConfigured();
}

size_t GeneticAlgorithmMinimizer::SetVariable(const std::string& parName,
        double initialValue, double initialStep) {

    size_t i = FitsMinimizerModule::SetVariable(parName, initialValue,
            initialStep);
    m_geneticAlgorithm->addUnlimitedVariable(parName, initialValue,
            initialStep);
    return i;
}

size_t GeneticAlgorithmMinimizer::SetLimitedVariable(const std::string& parName,
        double initialValue, double initialStep,
        const std::pair<double, double>& limits) {

    size_t i = FitsMinimizerModule::SetLimitedVariable(parName, initialValue,
            initialStep, limits);
    m_geneticAlgorithm->addLimitedVariable(parName, initialValue, limits,
            initialStep);
    return i;
}

size_t GeneticAlgorithmMinimizer::SetFixedVariable(const std::string& parName,
        double initialValue) {

    warn(__func__, "Initial value ignored");

    size_t i = FitsMinimizerModule::SetFixedVariable(parName, initialValue);
    m_geneticAlgorithm->addFixedVariable(parName, initialValue);
    return i;
}

double GeneticAlgorithmMinimizer::getVariableValue(size_t i) const {

    if (i >= getNVariables()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Illegal index");
    }

    return (m_geneticAlgorithm->getBestParameters()).at(i);
}

double GeneticAlgorithmMinimizer::getVariableValue(
        const std::string& varName) const {
    return getVariableValue(getVariableIndex(varName));
}

double GeneticAlgorithmMinimizer::getVariableError(size_t i) const {

    warn(__func__, "Not implemented");
    return 0.;
}

double GeneticAlgorithmMinimizer::getVariableError(
        const std::string& varName) const {

    warn(__func__, "Not implemented");
    return 0.;
}

double GeneticAlgorithmMinimizer::getFCNValue() const {
    return m_geneticAlgorithm->getBestFcn();
}

FitStatusType::Type GeneticAlgorithmMinimizer::getFitStatus() const {
    return m_geneticAlgorithm->getFitStatus();
}

size_t GeneticAlgorithmMinimizer::getNCalls() const {
    return m_geneticAlgorithm->getNCalls();
}

void GeneticAlgorithmMinimizer::runMinimization() {
    m_geneticAlgorithm->run();
}

void GeneticAlgorithmMinimizer::setFCN(FitsService* c, fcnType fcn,
        size_t nPar) {
    m_geneticAlgorithm->setFCN(c, fcn);
}

NumA::MatrixD GeneticAlgorithmMinimizer::getCovarianceMatrix() const {

    warn(__func__, "No implemented yet");
    return NumA::MatrixD(0, 0);
}

NumA::MatrixD GeneticAlgorithmMinimizer::getCorrelationMatrix() const {

    warn(__func__, "No implemented yet");
    return NumA::MatrixD(0, 0);
}
