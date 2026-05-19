/**
 * @file DVCSAluMinusSin1PhiTorch.cpp
 *
 * Implements DVCSAluMinusSin1PhiTorch::compute() using 10-point
 * Gauss–Legendre quadrature over φ ∈ [0,2π].
 *
 * The GL nodes on [–1,1] were obtained from scipy.special.p_roots(10):
 *
 *   from scipy.special import p_roots
 *   nodes, weights = p_roots(10)
 *
 * and are hardcoded below as static arrays.  The mapping to [0,2π] is
 *
 *   φₖ = π · (1 + uₖ),   dφ = π · duₖ
 *
 * so that the GL sum approximates
 *
 *   ∫₀^{2π} f(φ) dφ  ≈  π · Σₖ wₖ f(φₖ)
 *
 * and therefore:
 *
 *   A_LU^{sin1φ} = (1/π) ∫₀^{2π} sin(φ) A_LU^–(φ) dφ
 *               ≈  Σₖ wₖ sin(φₖ) A_LU^–(φₖ)
 *
 * All CFF evaluations are performed with gradient tracking disabled
 * for the NN forward pass itself (the gradients needed for training flow
 * through the CFF output tensors that are kept in-graph for the
 * subsequent physics calculation).
 */

#include "../../../include/NNFit/theory/DVCSAluMinusSin1PhiTorch.h"
#include <cmath>
#include <stdexcept>

namespace Theory {

// ─────────────────────────────────────────────────────────────────────────────
// 10-point Gauss–Legendre nodes and weights on [–1, 1]
// (from scipy.special.p_roots(10), converted to double)
// ─────────────────────────────────────────────────────────────────────────────

// Nodes are symmetric about 0; listed from –1 side to +1 side
const double DVCSAluMinusSin1PhiTorch::s_gl_nodes[10] = {
    -0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
     0.1488743389816312,
     0.4333953941292472,
     0.6794095682990244,
     0.8650633666889845,
     0.9739065285171717
};

const double DVCSAluMinusSin1PhiTorch::s_gl_weights[10] = {
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159820,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159820,
    0.1494513491505806,
    0.0666713443086881
};

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

DVCSAluMinusSin1PhiTorch::DVCSAluMinusSin1PhiTorch(
    double xB, double t, double Q2, double E,
    CFFNNModel                      net,
    const std::vector<std::string>& outputLabels)
    : m_kin(computeKinematics(xB, t, Q2, E))
    , m_net(std::move(net))
    , m_outputLabels(outputLabels)
{}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: find label index
// ─────────────────────────────────────────────────────────────────────────────

int DVCSAluMinusSin1PhiTorch::labelIndex(const std::string& name) const
{
    for (int k = 0; k < static_cast<int>(m_outputLabels.size()); ++k)
        if (m_outputLabels[k] == name) return k;
    return -1;
}

torch::Tensor DVCSAluMinusSin1PhiTorch::extractCFF(
    const torch::Tensor& nnOut, const std::string& label) const
{
    int idx = labelIndex(label);
    if (idx < 0)
        return torch::zeros({}, nnOut.options());
    return nnOut[0][idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// compute()
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor DVCSAluMinusSin1PhiTorch::compute()
{
    // ── 1. Run NN once for (xB, t, Q2) ─────────────────────────────────────
    //    The NN takes a 1×3 input; gradients flow through its output.
    //    We use eval() mode but keep grad tracking on the *output* tensors.
    const double xB_d = m_kin.xB;
    // ξ = 2xB/(1+xB)  [or equivalently computed from ξ stored in module;
    // here we just use xB directly as the NN was trained on (xB, t, Q2)]
    // Input: [xB, t, Q2]
    torch::Tensor input = torch::zeros({1, 3});
    input[0][0] = static_cast<float>(xB_d);
    input[0][1] = static_cast<float>(m_kin.t);
    input[0][2] = static_cast<float>(m_kin.Q2);

    // Forward pass – switch to eval mode but keep autograd for outputs
    m_net->eval();
    torch::Tensor nnOut = m_net->forward(input);

    // ── 2. Extract the 8 CFF components ─────────────────────────────────────
    const torch::Tensor H_re  = extractCFF(nnOut, "ReH");
    const torch::Tensor H_im  = extractCFF(nnOut, "ImH");
    const torch::Tensor E_re  = extractCFF(nnOut, "ReE");
    const torch::Tensor E_im  = extractCFF(nnOut, "ImE");
    const torch::Tensor Ht_re = extractCFF(nnOut, "ReHt");
    const torch::Tensor Ht_im = extractCFF(nnOut, "ImHt");
    const torch::Tensor Et_re = extractCFF(nnOut, "ReEt");
    const torch::Tensor Et_im = extractCFF(nnOut, "ImEt");

    // ── 3. Dress the CFFs with kinematic helicity-combination coefficients ──
    DressedCFFs dc = computeDressedCFFs(
        m_kin,
        H_re, H_im, E_re, E_im, Ht_re, Ht_im, Et_re, Et_im);

    // ── 4. Pre-compute φ-independent Fourier coefficients ───────────────────
    VCSCoeffs    vcs    = computeVCSCoeffs(m_kin, dc);
    InterfCoeffs interf = computeInterfCoeffs(m_kin, dc);

    // ── 5. GL quadrature over φ ∈ [0, 2π] ──────────────────────────────────
    // Initialise accumulator as a zero torch scalar connected to the graph
    torch::Tensor result = torch::zeros({}, H_re.options());

    for (int k = 0; k < 10; ++k) {
        const double phi_k = PI * (1. + s_gl_nodes[k]);   // map [–1,1] → [0,2π]
        const double w_k   = s_gl_weights[k];

        // σ(+1, φₖ) and σ(–1, φₖ)
        torch::Tensor sig_p = crossSectionAtPhi(m_kin, vcs, interf, phi_k, +1.);
        torch::Tensor sig_m = crossSectionAtPhi(m_kin, vcs, interf, phi_k, -1.);

        torch::Tensor denom = sig_p + sig_m;

        // Guard against zero denominator (set A_LU = 0 if denominator ≈ 0)
        torch::Tensor alu_k = torch::where(
            denom.abs() > 1e-30,
            (sig_p - sig_m) / denom,
            torch::zeros_like(denom));

        // Accumulate:  Σₖ wₖ · sin(φₖ) · A_LU^–(φₖ)
        result = result + w_k * std::sin(phi_k) * alu_k;
    }

    return result;
}

} // namespace Theory
