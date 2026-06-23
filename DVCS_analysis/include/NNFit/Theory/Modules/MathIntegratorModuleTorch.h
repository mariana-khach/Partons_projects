//
// Created by Mariana Khachatryan on 6/15/26.
//

#ifndef MATH_INTEGRATOR_MODULE_TORCH_H
#define MATH_INTEGRATOR_MODULE_TORCH_H

#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <torch/torch.h>

#include <functional>

namespace NumA {
class Integrator1D;
} /* namespace NumA */

/**
 * @class MathIntegratorModuleTorch
 *
 * @brief libtorch counterpart of PARTONS::MathIntegratorModule.
 *
 * Owns a NumA integrator (selected via NumA::IntegratorType1D, exactly like
 * the scalar base class) and exposes integrateTorch(), which evaluates a
 * tensor-returning integrand at the quadrature nodes and returns the integral
 * as a torch::Tensor with the autograd graph intact.
 *
 * The trick in every supported case is the same: the quadrature reduces to a
 * weighted sum \f$ \int_a^b f(x)\,dx = \sum_i w_i\, f(x_i) \f$ whose nodes
 * \f$x_i\f$ and weights \f$w_i\f$ are constants (no dependence on the
 * integrand's parameters). Building that sum in tensor space therefore lets
 * gradients flow entirely through \f$f(x_i)\f$ — from the integrand's
 * parameters (e.g. NN weights) to the returned integral.
 *
 * Intended to be inherited by a tensor observable module, the way
 * DVCSAluMinusSin1Phi inherits PARTONS::MathIntegratorModule for the scalar
 * path, so setIntegrator()/integrateTorch() are available as protected
 * members.
 *
 * Supported integrator types (matched bit-for-bit against the corresponding
 * NumA scalar routine):
 *  - TRAPEZOIDAL, GL          — fixed-rule quadrature; nodes/weights read from
 *                               NumA::QuadratureIntegrator1D, linear remap.
 *  - TRAPEZOIDALLOG           — log-spaced trapezoid (requires a, b > 0).
 *  - DEXP                     — adaptive tanh-sinh (double exponential), the
 *                               default integrator across PARTONS observables.
 *                               The refinement level is selected exactly as in
 *                               NumA from detached values; the final estimate
 *                               is a fixed weighted sum, so gradients survive.
 *
 * @note GK21_ADAPTIVE is NOT supported: its Wynn epsilon extrapolation makes
 *       the result a nonlinear function of the (parameter-dependent) partial
 *       sums, which cannot be written as a constant-weight sum. integrateTorch()
 *       throws if it is configured. (PARTONS itself never selects GK21.)
 */
class MathIntegratorModuleTorch {
public:
    virtual ~MathIntegratorModuleTorch();

protected:
    MathIntegratorModuleTorch();
    MathIntegratorModuleTorch(const MathIntegratorModuleTorch& other);

    /**
     * Select a NumA integrator. Mirrors PARTONS::MathIntegratorModule::setIntegrator().
     * @param integratorType One of TRAPEZOIDAL, TRAPEZOIDALLOG, GL, DEXP.
     * @param nNodes For fixed-rule quadratures (TRAPEZOIDAL, GL): number of nodes
     *               to use (calls QuadratureIntegrator1D::setN). Ignored (0) for
     *               adaptive rules (DEXP); throws if > 0 for a non-quadrature rule.
     */
    void setIntegrator(NumA::IntegratorType1D::Type integratorType,
            unsigned int nNodes = 0);

    /**
     * Differentiable integral over [a, b].
     *
     * The integrand is evaluated batched: pFunction receives the [N] tensor of
     * node positions already mapped onto [a, b] and must return an [N] tensor of
     * the corresponding values (carrying the autograd graph). For DEXP the
     * integrand may be called once per refinement level (up to 10 times), each
     * call batched over that level's nodes; for the fixed rules it is called
     * exactly once.
     *
     * @param pFunction Integrand, x[N] -> f(x)[N]. Nodes are float64; the
     *                  returned tensor sets the dtype/device of the result.
     * @param a Lower bound.
     * @param b Upper bound.
     * @return 0-d torch::Tensor holding the integral, grad-connected to f.
     */
    torch::Tensor integrateTorch(
            const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
            double a, double b) const;

    /**
     * @return Pointer to the underlying NumA integrator (may be null). Exposed
     *         so callers can tune N / tolerances via the usual NumA interface.
     */
    NumA::Integrator1D* getMathIntegrator() const;

private:
    /** Fixed-rule path (TRAPEZOIDAL, GL): linear remap of NumA nodes/weights. */
    torch::Tensor integrateTorchQuadrature(
            const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
            double a, double b) const;

    /** Log-spaced trapezoid path (TRAPEZOIDALLOG); requires a, b > 0. */
    torch::Tensor integrateTorchTrapezoidalLog(
            const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
            double a, double b) const;

    /** Adaptive tanh-sinh path (DEXP). */
    torch::Tensor integrateTorchDExp(
            const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
            double a, double b) const;

    NumA::Integrator1D* m_mathIntegrator;          ///< Owned NumA integrator.
    NumA::IntegratorType1D::Type m_integratorType; ///< Selected type (drives dispatch).
};

#endif /* MATH_INTEGRATOR_MODULE_TORCH_H */
