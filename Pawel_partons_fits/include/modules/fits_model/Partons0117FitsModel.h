/*
 * Partons0117FitsModel.h
 *
 *  Created on: 17 Jun 2017
 *      Author: Pawel Sznajder (NCBJ)
 */

#ifndef PARTONS0117SMODEL_H_
#define PARTONS0117SMODEL_H_

#include <ElementaryUtils/parameters/Parameters.h>
#include <string>

#include "FitsModelModule.h"

class Partons0117BorderFunctionModel;
class Partons0117SubtractionConstantModel;

class Partons0117FitsModel: public FitsModelModule {

public:

    static const unsigned int classId;

    Partons0117FitsModel(const std::string& className);
    virtual ~Partons0117FitsModel();

    virtual void resolveObjectDependencies();
    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual Partons0117FitsModel* clone() const;

    virtual void initParameters();
    virtual void updateParameters(const double* Var);

    virtual void initModule();
    virtual void isModuleWellConfigured();

protected:

    Partons0117FitsModel(const Partons0117FitsModel &other);

private:

    double getSkewVal(double delta) const;
    double getSkewSea(double delta) const;

    Partons0117BorderFunctionModel* m_pPartons0117BorderFunctionModel;
    Partons0117SubtractionConstantModel* m_pPartons0117SubtractionConstantModel;
};

#endif /* PARTONS0117SMODEL_H_ */
