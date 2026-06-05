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
     * Bundle of the four leading-twist CFFs as 0-d torch::Tensors, in the
     * same (Re, Im) pairing accepted by Theory::computeDressedCFFs.
     * Missing labels in m_outputLayer become torch::zeros({}) (no grad).
     */
    struct AllCFFsTensor {
        torch::Tensor H_re,  H_im;
        torch::Tensor E_re,  E_im;
        torch::Tensor Ht_re, Ht_im;
        torch::Tensor Et_re, Et_im;
    };

    /**
     * Single NN forward pass at the cached (m_xi, m_t, m_Q2). Returns
     * the eight CFF components in one struct, sliced from a single
     * [1, n_outputs] forward result. Use this in the tensor pipeline
     * instead of calling computeCFFTensor(type) four times — same final
     * tensors, but 1× rather than 4× the NN cost. Same mode handling as
     * computeCFFTensor: caller controls train()/eval(), no NoGradGuard.
     */
    AllCFFsTensor computeAllCFFsTensor();

    /**
     * Push the kinematic state used by computeCFFTensor() / computeCFF()
     * onto the inherited protected members of DVCSConvolCoeffFunctionModule
     * (m_xi, m_t, m_Q2). PARTONS' scalar pipeline normally does this as a
     * side effect of computeConvolCoeffFunction(); this helper exposes the
     * same setup for callers that drive the tensor path directly and want
     * to avoid the discarded NN forward pass that a full computeCFF() call
     * would entail.
     *
     * @param xi GPD skewness (already converted from xB by the xi converter).
     * @param t  Mandelstam t (GeV²).
     * @param Q2 Photon virtuality (GeV²).
     */
    void setupKinematics(double xi, double t, double Q2);

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
