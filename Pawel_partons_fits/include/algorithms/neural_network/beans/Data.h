/*
 * NeuralNetworkData.h
 *
 *  Created on: May 1, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURALNETWORKDATA_H_
#define NEURALNETWORKDATA_H_

#include <partons/BaseObject.h>
#include <map>
#include <vector>

class Data: public PARTONS::BaseObject {

public:

    Data();
    Data(unsigned int size);
    Data(const Data& other);
    ~Data();
    Data* clone() const;

    unsigned int getNVariables() const;
    unsigned int getNPoints() const;
    const std::map<unsigned int, std::vector<double> >& getData() const;
    void addData(const std::vector<double>& data);

    void print() const;

    void operator+=(const Data& rhs);

private:

    unsigned int m_nVariables;
    unsigned int m_nPoints;
    std::map<unsigned int, std::vector<double> > m_data;
};

#endif /* NEURALNETWORKDATA_H_ */
