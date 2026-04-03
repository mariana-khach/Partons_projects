//
// Created by Mariana Khachatryan on 3/25/26.
//

#ifndef DVCS_CFF_NN_PYTORCH_H
#define DVCS_CFF_NN_PYTORCH_H

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>

#include <complex>
#include <map>
#include <string>
#include <vector>

#include "CFF_NN_Fit.h"

/**
 * @class DVCSCFFNNPytorch
 *
 * @brief DVCS CFFs evaluated from a libtorch-trained neural network.
 *
 * The trained CFFNNModel and output layer names are injected via setModel().
 * Input to the network: [xB, t, Q2] (3 features).
 * xB is derived from PARTONS m_xi as xB = 2*xi / (1 + xi).
 * If the output layer does not contain the Re or Im CFF name for the
 * requested GPD type, the corresponding value is set to 0.
 */
class DVCSCFFNNPytorch : public PARTONS::DVCSConvolCoeffFunctionModule {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    /**
     * Constructor.
     * @param className Name of last child class.
     */
    DVCSCFFNNPytorch(const std::string& className);

    virtual ~DVCSCFFNNPytorch();

    virtual DVCSCFFNNPytorch* clone() const;

    virtual void configure(const ElemUtils::Parameters& parameters);
    virtual void resolveObjectDependencies();
    virtual void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    virtual std::complex<double> computeCFF();

    /**
     * Inject the trained libtorch model and the output layer name list.
     * Must be called before the module is used for computation.
     * @param net     Trained CFFNNModel.
     * @param outputLayer  Names of each output neuron, e.g. {"ImH", "ReH"}.
     */
    void setModel(CFFNNModel net, const std::vector<std::string>& outputLayer);

protected:

    /**
     * Copy constructor.
     * @param other Object to be copied.
     */
    DVCSCFFNNPytorch(const DVCSCFFNNPytorch& other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    /**
     * Map GPDType to the CFF name suffix used in m_outputLayer.
     * GPDType::H -> "H", GPDType::E -> "E", GPDType::Ht -> "Ht", GPDType::Et -> "Et".
     */
    std::string gpdTypeToName(PARTONS::GPDType::Type type) const;

    CFFNNModel               m_net{nullptr};   ///< Trained libtorch model.
    std::vector<std::string> m_outputLayer;    ///< Output neuron names.
};

#endif /* DVCS_CFF_NN_PYTORCH_H */
