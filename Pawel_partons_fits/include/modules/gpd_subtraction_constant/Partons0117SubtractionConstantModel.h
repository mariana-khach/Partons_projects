/*
 * Partons0117SubtractionConstantModel.h
 *
 *  Created on: Jan 17, 2017
 *      Author: Pawel Sznajder (NCBJ)
 */

#ifndef PARTONS0117SUBTRACTIONCONSTANTMODEL_H_
#define PARTONS0117SUBTRACTIONCONSTANTMODEL_H_

#include <partons/modules/gpd_subtraction_constant/GPDSubtractionConstantModule.h>
#include <partons/utils/type/PhysicalType.h>
#include <string>

class Partons0117SubtractionConstantModel: public PARTONS::GPDSubtractionConstantModule {

public:

    static const unsigned int classId;
    Partons0117SubtractionConstantModel(const std::string &className);
    virtual ~Partons0117SubtractionConstantModel();
    virtual Partons0117SubtractionConstantModel* clone() const;

    double getParC() const;
    void setParC(double parC);
    double getParA() const;
    void setParA(double parA);

protected:

    Partons0117SubtractionConstantModel(
            const Partons0117SubtractionConstantModel& other);

    virtual PARTONS::PhysicalType<double> computeSubtractionConstant();

private:

    double m_par_C;
    double m_par_a;
};

#endif /* PARTONS0117SUBTRACTIONCONSTANTMODEL_H_ */
