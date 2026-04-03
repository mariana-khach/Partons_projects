#ifndef DVCS_CFF_DISPERSION_FOR_H_LIKE_STANDARD_FOR_E_LIKE_H
#define DVCS_CFF_DISPERSION_FOR_H_LIKE_STANDARD_FOR_E_LIKE_H

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <complex>
#include <map>
#include <string>

namespace PARTONS {
class GPDSubtractionConstantModule;
} /* namespace PARTONS */

class DVCSCFFDispersionForHLikeStandardForELike: public PARTONS::DVCSConvolCoeffFunctionModule {

public:

    static const unsigned int classId;

    DVCSCFFDispersionForHLikeStandardForELike(const std::string &className);
    virtual DVCSCFFDispersionForHLikeStandardForELike* clone() const;
    virtual ~DVCSCFFDispersionForHLikeStandardForELike();

    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual void resolveObjectDependencies();

    virtual void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    virtual std::complex<double> computeCFF();

    PARTONS::DVCSConvolCoeffFunctionModule* getDVCSCFFDispersionRelationModel() const;
    void setDVCSCFFDispersionRelationModel(
            PARTONS::DVCSConvolCoeffFunctionModule* dvcsCFFDispersionRelationModel);

    PARTONS::DVCSConvolCoeffFunctionModule* getDVCSCFFStandardModel() const;
    void setDVCSCFFStandardModel(
            PARTONS::DVCSConvolCoeffFunctionModule* dvcsCFFStandardModel);

    PARTONS::GPDSubtractionConstantModule* getSubtractionConstantModule() const;
    void setSubtractionConstantModule(
            PARTONS::GPDSubtractionConstantModule* subtractionConstantModule);

protected:

    DVCSCFFDispersionForHLikeStandardForELike(
            const DVCSCFFDispersionForHLikeStandardForELike &other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    std::complex<double> computeCFFHHt();
    std::complex<double> computeCFFEEt();

    PARTONS::DVCSConvolCoeffFunctionModule* m_pDVCSCFFDispersionRelationModel;
    PARTONS::DVCSConvolCoeffFunctionModule* m_pDVCSCFFStandardModel;
    PARTONS::GPDSubtractionConstantModule* m_pSubtractionConstantModule;
};

#endif /* DVCS_CFF_DISPERSION_FOR_H_LIKE_STANDARD_FOR_E_LIKE_H */
