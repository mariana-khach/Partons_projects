/*
 * NeuralNetworkData.cpp
 *
 *  Created on: May 1, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/Data.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <cstdio>
#include <utility>

Data::Data() :
        BaseObject("NeuralNetworkInput") {

    m_nVariables = 0;
    m_nPoints = 0;
    m_data.clear();
}

Data::Data(unsigned int size) :
        BaseObject("NeuralNetworkInput") {

    if (size == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Input vector is empty");
    }

    m_nVariables = size;
    m_nPoints = 0;

    for (unsigned int i = 0; i < size; i++) {
        m_data.insert(std::make_pair(i, std::vector<double>()));
    }
}

Data::Data(const Data& other) :
        BaseObject(other) {

    m_nVariables = other.m_nVariables;
    m_nPoints = other.m_nPoints;
    m_data = other.m_data;
}

Data::~Data() {
}

Data* Data::clone() const {
    return new Data(*this);
}

unsigned int Data::getNVariables() const {
    return m_nVariables;
}

unsigned int Data::getNPoints() const {
    return m_nPoints;
}

const std::map<unsigned int, std::vector<double> >& Data::getData() const {
    return m_data;
}

void Data::addData(const std::vector<double>& data) {

    if (data.size() != m_nVariables) {

        throw ElemUtils::CustomException(getClassName(), __func__,
                "Size of input vector different than that declared in the constructor");
    }

    std::map<unsigned int, std::vector<double> >::iterator it;

    for (unsigned int i = 0; i < data.size(); i++) {

        it = m_data.find(i);

        if (it == m_data.end()) {

            throw ElemUtils::CustomException(getClassName(), __func__,
                    "Map containing input values inconsistent");
        }

        (it->second).push_back(data[i]);
    }

    m_nPoints++;
}

void Data::print() const {

    printf("Number of variables: %u\n", m_nVariables);
    printf("Number of entries: %u\n", m_nPoints);

    for (unsigned int i = 0; i < m_nPoints; i++) {

        printf("\tEntry: %4u\tVariables: ", i);

        for (unsigned int j = 0; j < m_nVariables; j++) {
            printf("\t%.4E", (m_data.find(j)->second).at(i));
        }

        printf("\n");
    }
}

void Data::operator+=(const Data& rhs) {

    if (rhs.getNVariables() != m_nVariables) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "RHS object defined for different number of variables");
    }

    std::map<unsigned int, std::vector<double> >::iterator itTarget;
    std::map<unsigned int, std::vector<double> >::const_iterator itSource;

    for (unsigned int i = 0; i < m_nVariables; i++) {

        itTarget = m_data.find(i);
        itSource = rhs.getData().find(i);

        for (unsigned int j = 0; j < rhs.getNPoints(); j++) {

            (itTarget->second).push_back((itSource->second).at(j));
        }
    }

    m_nPoints += rhs.getNPoints();
}

