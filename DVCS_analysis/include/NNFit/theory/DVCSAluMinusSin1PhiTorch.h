#pragma once

/**
 * @file DVCSAluMinusSin1PhiTorch.h
 *
 * Fully differentiable computation of the DVCS beam-spin asymmetry
 *
 *   A_LU^{sin1φ} = (1/π) ∫₀^{2π} dφ sin(φ) A_LU^–(φ)
 *
 * where  A_LU^–(φ) = [σ(λ=+1,φ) – σ(λ=–1,φ)] / [σ(λ=+1,φ) + σ(λ=–1,φ)]
 *
 * implemented using 10-point Gauss–Legendre quadrature so that the φ
 * integral stays entirely inside the torch autograd graph.
 *
 * Quadrature:
 *   nodes uₖ on [–1,1] (from scipy.special.p_roots(10)) are mapped to
 *   φₖ = π(1+uₖ) ∈ [0,2π] with weights wₖ, giving
 *
 *   A_LU^{sin1φ} ≈ Σₖ wₖ · sin(φₖ) · A_LU^–(φₖ)
 *
 * Usage:
 *   DVCSAluMinusSin1PhiTorch obs(xB, t, Q2, E, net, output_labels);
 *   torch::Tensor result = obs.compute();
 */

#include "DVCSKinematicsTorch.h"
#include "DVCSAmplitudesBMJ12Torch.h"
#include "../CFF_NN_Fit.h"     // for CFFNNModel typedef
#include <string>
#include <vector>

namespace Theory {

class DVCSAluMinusSin1PhiTorch
{
public:
    /**
     * @param xB           Bjorken x
     * @param t            Mandelstam t [GeV²]
     * @param Q2           Photon virtuality [GeV²]
     * @param E            Beam energy [GeV]
     * @param net          Trained CFF neural network (libtorch Module)
     * @param outputLabels Ordered list of NN output names, e.g.
     *                     {"ReH","ImH","ReE","ImE","ReHt","ImHt","ReEt","ImEt"}
     */
    DVCSAluMinusSin1PhiTorch(double xB, double t, double Q2, double E,
                             CFFNNModel                       net,
                             const std::vector<std::string>&  outputLabels);

    /**
     * Evaluate A_LU^{sin1φ} for the stored kinematics.
     * The returned scalar torch::Tensor carries gradients w.r.t. the NN
     * weights, enabling direct fitting to experimental data.
     *
     * @return  A_LU^{sin1φ}  as a 0-d torch::Tensor
     */
    torch::Tensor compute();

private:
    DVCSKin                  m_kin;
    CFFNNModel               m_net;
    std::vector<std::string> m_outputLabels;

    // 10-point Gauss–Legendre nodes and weights on [–1,1]
    // (equivalent to scipy.special.p_roots(10))
    static const double s_gl_nodes[10];
    static const double s_gl_weights[10];

    /// Find the column index of @p name in m_outputLabels; returns –1 if absent.
    int labelIndex(const std::string& name) const;

    /// Extract a single output component from the NN forward result.
    /// Returns a zero tensor if the label is not present.
    torch::Tensor extractCFF(const torch::Tensor& nnOut,
                             const std::string&   label) const;
};

} // namespace Theory