#include "../../../include/modules/gpd_subtraction_constant/Partons0117SubtractionConstantModel.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <cmath>

using namespace PARTONS;

const unsigned int Partons0117SubtractionConstantModel::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new Partons0117SubtractionConstantModel(
                        "Partons0117SubtractionConstantModel"));

Partons0117SubtractionConstantModel::Partons0117SubtractionConstantModel(
        const std::string& className) :
        GPDSubtractionConstantModule(className) {

    m_par_C = 0.;
    m_par_a = 0.;
}

Partons0117SubtractionConstantModel::~Partons0117SubtractionConstantModel() {
}

Partons0117SubtractionConstantModel* Partons0117SubtractionConstantModel::clone() const {
    return new Partons0117SubtractionConstantModel(*this);
}

Partons0117SubtractionConstantModel::Partons0117SubtractionConstantModel(
        const Partons0117SubtractionConstantModel& other) :
        GPDSubtractionConstantModule(other) {

    m_par_C = other.m_par_C;
    m_par_a = other.m_par_a;
}

PhysicalType<double> Partons0117SubtractionConstantModel::computeSubtractionConstant() {
    return PhysicalType<double>(m_par_C * exp(m_par_a * m_t), PhysicalUnit::NONE);
}

double Partons0117SubtractionConstantModel::getParC() const {
    return m_par_C;
}

void Partons0117SubtractionConstantModel::setParC(double parC) {
    m_par_C = parC;
}

double Partons0117SubtractionConstantModel::getParA() const {
    return m_par_a;
}

void Partons0117SubtractionConstantModel::setParA(double parA) {
    m_par_a = parA;
}
