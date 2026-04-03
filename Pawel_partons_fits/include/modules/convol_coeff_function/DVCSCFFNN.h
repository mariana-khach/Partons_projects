#ifndef DVCS_CFF_NN_H
#define DVCS_CFF_NN_H

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <complex>
#include <map>
#include <string>
#include <utility>

#include "../../algorithms/neural_network/neural_network/NeuralNetwork.h"

class NeuralNetwork;

class DVCSCFFNN: public PARTONS::DVCSConvolCoeffFunctionModule {

public:

    static const unsigned int classId;

    DVCSCFFNN(const std::string &className);
    virtual DVCSCFFNN* clone() const;
    virtual ~DVCSCFFNN();

    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual void resolveObjectDependencies();

    virtual void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    virtual std::complex<double> computeCFF();

    const std::map<PARTONS::GPDType::Type, NeuralNetwork*>& getNeuralNetworks() const;

protected:

    DVCSCFFNN(const DVCSCFFNN &other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    std::map<PARTONS::GPDType::Type, NeuralNetwork*> m_neuralNetworks;

    std::pair<double, double> m_rangeLog10Xi;
    std::pair<double, double> m_rangeT;
    std::pair<double, double> m_rangeQ2;
    std::map<PARTONS::GPDType::Type, std::pair<double, double> > m_rangeReCFF;
    std::map<PARTONS::GPDType::Type, std::pair<double, double> > m_rangeXiImCFF;

};

#endif /* DVCS_CFF_NN_H */
