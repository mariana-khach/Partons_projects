#ifndef CUSTOM_LOSS_H
#define CUSTOM_LOSS_H

/**
 * @file CustomLoss.h
 *
 * @brief χ² loss for training the CFF NN directly on DVCS observable data.
 *
 *   χ²  =  Σᵢ ( A_LU^{sin1φ}(NN(xᵢ)) − yᵢ )² / σᵢ²
 *
 * For each kinematic point xᵢ = (xBᵢ, tᵢ, Q²ᵢ, Eᵢ), A_LU^{sin1φ} is
 * evaluated through the PARTONS-registered tensor module chain
 *   DVCSAluMinusSin1PhiTorch → DVCSProcessBMJ12Torch → DVCSCFFNNPytorch,
 * keeping the autograd graph alive from the NN weights through the full
 * physics calculation back to a single 0-d loss tensor.
 *
 * Subclasses torch::nn::Module so it composes naturally with libtorch's
 * other primitives (registered submodules, parameter discovery, etc.) and
 * can be called like any PyTorch loss module:
 *
 *   CustomLoss loss(net, output_layer);     // construct once
 *   auto chi2 = loss(X, E, y_obs, sigma);   // per-batch, autograd-aware
 *   chi2.backward();
 *   optimizer.step();
 */

#include <torch/torch.h>

#include <string>
#include <vector>

#include "NNFit/CFF_NN_Fit.h"

// Forward declarations — these are the PARTONS-registered tensor modules
// the loss drives. Forward-declaring keeps PARTONS headers out of this
// header; only CustomLoss.cpp needs to know about them.
class DVCSCFFNNPytorch;
class DVCSProcessBMJ12Torch;
class DVCSAluMinusSin1PhiTorch;

struct CustomLossImpl : public torch::nn::Module {

    /**
     * Construct the loss. Instantiates the three PARTONS-tensor modules
     * via Partons::getInstance()->getModuleObjectFactory(), wires them
     * together, and injects the NN into the CFF module.
     *
     * Prerequisite: PARTONS must already be initialised
     * (Partons::getInstance()->init() has been called).
     *
     * @param net           Trained or in-training CFFNNModel — gradients
     *                      from chi².backward() flow back to its parameters.
     * @param output_layer  Names of NN output neurons (CFF labels).
     */
    CustomLossImpl(CFFNNModel net,
                   const std::vector<std::string>& output_layer);

    /**
     * Evaluate the χ² loss over a batch of data points.
     *
     * @param X     [N, 3] kinematics (xB, t, Q²) — raw values, no scaling.
     * @param E     [N]    beam energy per row.
     * @param y_obs [N]    measured observable value (A_LU^{sin1φ}).
     * @param sigma [N]    uncertainty on the observable.
     * @return χ² as a 0-d torch::Tensor connected to the autograd graph
     *         (gradients ∂χ²/∂(NN params) flow on .backward()).
     */
    torch::Tensor forward(const torch::Tensor& X,
                          const torch::Tensor& E,
                          const torch::Tensor& y_obs,
                          const torch::Tensor& sigma);

private:

    // The three PARTONS-tensor modules driving the differentiable
    // observable pipeline. Raw pointers because they are owned by the
    // PARTONS factory/registry, not by this loss.
    DVCSCFFNNPytorch*         m_pCFF;
    DVCSProcessBMJ12Torch*    m_pProcess;
    DVCSAluMinusSin1PhiTorch* m_pObs;
};

TORCH_MODULE(CustomLoss);

#endif /* CUSTOM_LOSS_H */