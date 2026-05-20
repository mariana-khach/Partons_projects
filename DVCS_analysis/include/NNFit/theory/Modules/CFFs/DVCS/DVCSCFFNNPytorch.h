//
// Created by Mariana Khachatryan on 3/25/26.
//

#ifndef DVCS_CFF_NN_PYTORCH_H
#define DVCS_CFF_NN_PYTORCH_H

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>

#include <torch/torch.h>

#include <complex>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "NNFit/CFF_NN_Fit.h"

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
     * Differentiable counterpart of computeCFF(): runs the NN forward pass
     * on the current PARTONS kinematics (m_xi, m_t, m_Q2) and returns the
     * Re and Im parts of the requested CFF as 0-d torch::Tensors that
     * remain connected to the autograd graph. The mode of the underlying
     * CFFNNModel is left untouched, so the caller controls train()/eval().
     *
     * If the requested GPD type's Re or Im name is absent from the output
     * layer, the corresponding tensor is a fresh torch::zeros({}) (no grad).
     *
     * @param gpdType GPD type whose Re/Im CFF is to be returned
     *                (H, E, Ht, Et). Independent of PARTONS dispatch
     *                state, so it can be called outside the PARTONS path.
     * @return pair (Re, Im), both 0-d tensors.
     */
    std::pair<torch::Tensor, torch::Tensor> computeCFFTensor(
            PARTONS::GPDType::Type gpdType);

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
