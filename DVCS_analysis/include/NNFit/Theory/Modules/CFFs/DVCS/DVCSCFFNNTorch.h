//
// Created by Mariana Khachatryan on 3/25/26.
//

#ifndef DVCS_CFF_NN_TORCH_H
#define DVCS_CFF_NN_TORCH_H

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>

#include <complex>
#include <map>
#include <string>
#include <vector>

#include "NNFit/CFF_NN_Fit.h"

/**
 * @class DVCSCFFNNTorch
 *
 * @brief DVCS CFFs evaluated from a libtorch-trained neural network.
 *
 * The trained CFFNNModel and output layer names are injected via setModel().
 * Input to the network: [xB, t, Q2] (3 features).
 * xB is derived from PARTONS m_xi as xB = 2*xi / (1 + xi).
 * If the output layer does not contain the Re or Im CFF name for the
 * requested GPD type, the corresponding value is set to 0.
 */
class DVCSCFFNNTorch : public PARTONS::DVCSConvolCoeffFunctionModule {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    /**
     * Constructor.
     * @param className Name of last child class.
     */
    DVCSCFFNNTorch(const std::string& className);

    virtual ~DVCSCFFNNTorch();

    virtual DVCSCFFNNTorch* clone() const;

    virtual void configure(const ElemUtils::Parameters& parameters);
    virtual void resolveObjectDependencies();
    virtual void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    virtual std::complex<double> computeCFF();

    /**
     * The four standard DVCS CFFs as 0-d complex (float64) tensors, produced by
     * a single NN forward pass. Components the network does not output (no
     * "Re<name>"/"Im<name>" in the output layer) are zero.
     */
    struct AllCFFsTensor {
        torch::Tensor H;  ///< CFF H  (0-d complex double).
        torch::Tensor E;  ///< CFF E.
        torch::Tensor Ht; ///< CFF Ht (H-tilde).
        torch::Tensor Et; ///< CFF Et (E-tilde).
    };

    /**
     * Set the CFF-module kinematics for the tensor path (the scalar path sets
     * them through PARTONS' setKinematics, which is protected). Must be called
     * before computeAllCFFsTensor()/computeCFFTensor().
     * @param xi Skewness.
     * @param t  Mandelstam t (GeV^2).
     * @param Q2 Photon virtuality (GeV^2).
     */
    void setupKinematicsTorch(double xi, double t, double Q2);

    /**
     * One NN forward pass returning all four CFFs as complex tensors with the
     * autograd graph connected to the network parameters.
     */
    AllCFFsTensor computeAllCFFsTensor();

    /**
     * The CFF of a single GPD type as a 0-d complex tensor (grad-tracked).
     * Returns 0 for types the network does not output.
     */
    torch::Tensor computeCFFTensor(PARTONS::GPDType::Type type);

    /**
     * Inject the trained libtorch model, the output layer name list, and the
     * per-feature min-max scaling fitted on the training set.
     * Must be called before the module is used for computation.
     *
     * The scaling travels with the model: the network input [xB, t, Q2] is
     * transformed as (x - xMin) / (xMax - xMin) to match the scaling applied
     * during training. If xMin/xMax are left undefined (default), computeCFF()
     * feeds raw features — use this only when the model was trained unscaled.
     *
     * @param net          Trained CFFNNModel.
     * @param outputLayer  Names of each output neuron, e.g. {"ImH", "ReH"}.
     * @param xMin         Per-feature minima, shape [3]. Undefined = no scaling.
     * @param xMax         Per-feature maxima, shape [3]. Undefined = no scaling.
     */
    void setModel(CFFNNModel net, const std::vector<std::string>& outputLayer,
            const torch::Tensor& xMin = {}, const torch::Tensor& xMax = {});

protected:

    /**
     * Copy constructor.
     * @param other Object to be copied.
     */
    DVCSCFFNNTorch(const DVCSCFFNNTorch& other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    /**
     * Map GPDType to the CFF name suffix used in m_outputLayer.
     * GPDType::H -> "H", GPDType::E -> "E", GPDType::Ht -> "Ht", GPDType::Et -> "Et".
     */
    std::string gpdTypeToName(PARTONS::GPDType::Type type) const;

    /**
     * Run the NN once on [xB, t, Q2] (with optional min-max scaling) and return
     * the [1, Nout] output cast to float64. The autograd graph to the network
     * parameters is preserved (no NoGradGuard here — callers that want the
     * scalar value wrap their call in one).
     */
    torch::Tensor forwardNN();

    /**
     * Build a 0-d complex (float64) tensor from the named Re/Im output neurons,
     * or 0 if the network does not provide them.
     */
    torch::Tensor cffComponentTensor(const torch::Tensor& output,
            const std::string& name) const;

    CFFNNModel               m_net{nullptr};   ///< Trained libtorch model.
    std::vector<std::string> m_outputLayer;    ///< Output neuron names.
    torch::Tensor            m_xMin;           ///< Per-feature minima [3] (training set).
    torch::Tensor            m_xMax;           ///< Per-feature maxima [3] (training set).
};

#endif /* DVCS_CFF_NN_TORCH_H */
