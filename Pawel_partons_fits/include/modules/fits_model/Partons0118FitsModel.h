/*
 * Partons0118FitsModel.h
 *
 *  Created on: 17 Jun 2017
 *      Author: Pawel Sznajder (NCBJ)
 */

#ifndef PARTONS0118SMODEL_H_
#define PARTONS0118SMODEL_H_

#include <ElementaryUtils/parameters/Parameters.h>
#include <string>

#include "FitsModelModule.h"

class Partons0118BorderFunctionModel;
class Partons0118SubtractionConstantModel;

class Partons0118FitsModel: public FitsModelModule {

public:

    static const unsigned int classId;

    Partons0118FitsModel(const std::string& className);
    virtual ~Partons0118FitsModel();

    virtual void resolveObjectDependencies();
    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual Partons0118FitsModel* clone() const;

    virtual void initParameters();
    virtual void updateParameters(const double* Var);

    virtual void initModule();
    virtual void isModuleWellConfigured();

protected:

    Partons0118FitsModel(const Partons0118FitsModel &other);

private:

//    double getSkewVal(double delta) const;
//    double getSkewSea(double delta) const;

    Partons0118BorderFunctionModel* m_pPartons0118BorderFunctionModel;
    Partons0118SubtractionConstantModel* m_pPartons0118SubtractionConstantModel;
};

#endif /* PARTONS0118SMODEL_H_ */
