//
// Created by Mariana Khachatryan on 6/15/26.
//

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"

#include <NumA/integration/one_dimension/Integrator1D.h>
#include <NumA/integration/one_dimension/QuadratureIntegrator1D.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

const torch::TensorOptions kF64 = torch::TensorOptions().dtype(torch::kFloat64);

// ---------------------------------------------------------------------------
// DEXP (double exponential / tanh-sinh) quadrature tables.
//
// NumA::DExpIntegrator1D hard-codes ~1500 nodes/weights in its .cpp. They are
// exactly the tanh-sinh rule
//     node(t)   = tanh( (pi/2) sinh(t) )
//     weight(t) = (pi/2) cosh(t) / cosh^2( (pi/2) sinh(t) )
// (verified to match the stored table to full double precision), so we
// regenerate them from the closed form rather than copy the table. The level
// structure mirrors NumA exactly:
//   - a single centre node t = 0,
//   - level 0 block: t = 1, 2, 3 (step h = 1),
//   - level l >= 1 block: the new midpoints t = (2k-1) * 2^-l with t < 3.
// (NumA offsets {1,4,7,13,25,49,97,193,385,769,1537} reproduce these counts.)
// ---------------------------------------------------------------------------

struct DExpTables {
    double centreWeight;                       ///< weight of the t = 0 node (pi/2).
    std::vector<torch::Tensor> nodes;          ///< nodes[block]: reference nodes in (0,1).
    std::vector<torch::Tensor> weights;        ///< weights[block]: matching weights.
    // block 0 = level-0 extra points {1,2,3}; block l (l>=1) = level l midpoints.
};

const DExpTables& dExpTables() {
    static const DExpTables tables = [] {
        const double pi2 = 0.5 * M_PI;
        auto node = [&](double t) { return std::tanh(pi2 * std::sinh(t)); };
        auto weight = [&](double t) {
            double ch = std::cosh(pi2 * std::sinh(t));
            return pi2 * std::cosh(t) / (ch * ch);
        };

        DExpTables t;
        t.centreWeight = pi2; // weight(0)

        // level-0 block: t = 1, 2, 3
        {
            std::vector<double> n, w;
            for (double tt : {1.0, 2.0, 3.0}) {
                n.push_back(node(tt));
                w.push_back(weight(tt));
            }
            t.nodes.push_back(torch::tensor(n, kF64));
            t.weights.push_back(torch::tensor(w, kF64));
        }

        // levels 1..9: new midpoints t = (2k-1) * h, h = 2^-level, t < 3
        for (int level = 1; level <= 9; ++level) {
            const double h = std::ldexp(1.0, -level);
            std::vector<double> n, w;
            for (int k = 1;; k += 2) {
                const double tt = k * h;
                if (tt >= 3.0) {
                    break;
                }
                n.push_back(node(tt));
                w.push_back(weight(tt));
            }
            t.nodes.push_back(torch::tensor(n, kF64));
            t.weights.push_back(torch::tensor(w, kF64));
        }
        return t;
    }();
    return tables;
}

} // namespace

MathIntegratorModuleTorch::MathIntegratorModuleTorch() :
        m_mathIntegrator(0), m_integratorType(NumA::IntegratorType1D::UNDEFINED) {
}

MathIntegratorModuleTorch::~MathIntegratorModuleTorch() {
    if (m_mathIntegrator) {
        delete m_mathIntegrator;
        m_mathIntegrator = 0;
    }
}

MathIntegratorModuleTorch::MathIntegratorModuleTorch(
        const MathIntegratorModuleTorch& other) :
        m_integratorType(other.m_integratorType) {
    if (other.m_mathIntegrator) {
        m_mathIntegrator = other.m_mathIntegrator->clone();
    } else {
        m_mathIntegrator = 0;
    }
}

void MathIntegratorModuleTorch::setIntegrator(
        NumA::IntegratorType1D::Type integratorType, unsigned int nNodes) {
    if (m_mathIntegrator) {
        delete m_mathIntegrator;
        m_mathIntegrator = 0;
    }
    m_mathIntegrator = NumA::Integrator1D::newIntegrator(integratorType);
    m_integratorType = integratorType;

    // For fixed-rule quadratures, set the number of nodes (computes nodes/weights).
    if (nNodes > 0) {
        NumA::QuadratureIntegrator1D* quad =
                dynamic_cast<NumA::QuadratureIntegrator1D*>(m_mathIntegrator);
        if (!quad) {
            throw std::runtime_error(
                    "MathIntegratorModuleTorch::setIntegrator: nNodes only applies "
                    "to fixed-rule quadratures (TRAPEZOIDAL, GL).");
        }
        quad->setN(nNodes);
    }
}

NumA::Integrator1D* MathIntegratorModuleTorch::getMathIntegrator() const {
    return m_mathIntegrator;
}

torch::Tensor MathIntegratorModuleTorch::integrateTorch(
        const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
        double a, double b) const {

    if (!m_mathIntegrator) {
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorch: no integrator set; "
                "call setIntegrator() first.");
    }

    switch (m_integratorType) {
    case NumA::IntegratorType1D::TRAPEZOIDAL:
    case NumA::IntegratorType1D::GL:
        return integrateTorchQuadrature(pFunction, a, b);
    case NumA::IntegratorType1D::TRAPEZOIDALLOG:
        return integrateTorchTrapezoidalLog(pFunction, a, b);
    case NumA::IntegratorType1D::DEXP:
        return integrateTorchDExp(pFunction, a, b);
    case NumA::IntegratorType1D::GK21_ADAPTIVE:
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorch: GK21_ADAPTIVE is not "
                "supported. Its Wynn epsilon extrapolation makes the result a "
                "nonlinear function of parameter-dependent partial sums, which "
                "cannot be expressed as a constant-weight quadrature. Use GL, "
                "TRAPEZOIDAL or DEXP.");
    default:
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorch: unsupported / "
                "undefined integrator type.");
    }
}

// Fixed-rule quadrature: matches NumA::QuadratureIntegrator1D::integrate(), which
// maps the reference nodes (on [-1,1]) onto [a,b] via x = d + c*node and returns
// c * sum_i w_i f(x_i), with c = (b-a)/2, d = (a+b)/2.
torch::Tensor MathIntegratorModuleTorch::integrateTorchQuadrature(
        const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
        double a, double b) const {

    NumA::QuadratureIntegrator1D* quad =
            dynamic_cast<NumA::QuadratureIntegrator1D*>(m_mathIntegrator);
    if (!quad) {
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorchQuadrature: integrator "
                "is not a fixed-rule quadrature.");
    }

    const std::vector<double>& refNodes = quad->getNodes();
    const std::vector<double>& refWeights = quad->getWeights();
    if (refNodes.empty() || refWeights.size() != refNodes.size()) {
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorchQuadrature: empty or "
                "inconsistent nodes/weights; set the number of nodes first.");
    }

    const double c = 0.5 * (b - a);
    const double d = 0.5 * (a + b);

    torch::Tensor nodes = torch::tensor(refNodes, kF64);
    torch::Tensor weights = torch::tensor(refWeights, kF64);
    torch::Tensor x = d + c * nodes;

    torch::Tensor fvals = pFunction(x);
    return (weights.to(fvals.options()) * fvals).sum() * c;
}

// Log-spaced trapezoid: matches NumA::TrapezoidalLogIntegrator1D::integrate().
// Points are x_i = exp(logA + i*logStep); the (non-uniform) trapezoid rule is a
// fixed weighted sum sum_i w_i f(x_i) with w_0, w_{N-1} half-steps and interior
// weights half the surrounding span. Requires a, b > 0.
torch::Tensor MathIntegratorModuleTorch::integrateTorchTrapezoidalLog(
        const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
        double a, double b) const {

    if (a <= 0. || b <= 0.) {
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorchTrapezoidalLog: range "
                "must be > 0.");
    }

    NumA::QuadratureIntegrator1D* quad =
            dynamic_cast<NumA::QuadratureIntegrator1D*>(m_mathIntegrator);
    if (!quad) {
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorchTrapezoidalLog: "
                "integrator is not a quadrature with a node count.");
    }
    const unsigned int N = quad->getN();
    if (N < 2) {
        throw std::runtime_error(
                "MathIntegratorModuleTorch::integrateTorchTrapezoidalLog: need "
                "at least 2 nodes.");
    }

    const double logA = std::log(a);
    const double logStep = (std::log(b) - logA) / double(N - 1);

    std::vector<double> xs(N), w(N, 0.);
    for (unsigned int i = 0; i < N; ++i) {
        xs[i] = (i == 0) ? a : std::exp(logA + i * logStep);
    }
    for (unsigned int i = 1; i < N; ++i) {
        const double dx = 0.5 * (xs[i] - xs[i - 1]);
        w[i - 1] += dx;
        w[i] += dx;
    }

    torch::Tensor x = torch::tensor(xs, kF64);
    torch::Tensor weights = torch::tensor(w, kF64);

    torch::Tensor fvals = pFunction(x);
    return (weights.to(fvals.options()) * fvals).sum();
}

// Adaptive tanh-sinh (double exponential). Reproduces the exact accumulation of
// NumA::DExpIntegrator1D::integrate(): start from the centre + level-0 estimate,
// then for each level halve the running step h, add that level's symmetric-pair
// contribution times h, and mix integral <- 0.5*integral + newContribution. The
// refinement level is chosen by the same convergence test as NumA, evaluated on
// detached values; the running `integral` is a tensor, so gradients flow through.
torch::Tensor MathIntegratorModuleTorch::integrateTorchDExp(
        const std::function<torch::Tensor(const torch::Tensor&)>& pFunction,
        double a, double b) const {

    const DExpTables& tab = dExpTables();

    const double c = 0.5 * (b - a);
    const double d = 0.5 * (a + b);

    // Build the node positions and weights for one symmetric-pair block, mapped
    // onto [a,b]: positions [d + c*n ... , d - c*n ...], weights [w ... , w ...].
    auto blockPositions = [&](const torch::Tensor& refNodes) {
        return torch::cat({d + c * refNodes, d - c * refNodes});
    };
    auto blockWeights = [&](const torch::Tensor& refWeights) {
        return torch::cat({refWeights, refWeights});
    };

    // Initial estimate: centre node (t=0, single) plus the level-0 block.
    torch::Tensor pos0 = torch::cat(
            {torch::full({1}, d, kF64), blockPositions(tab.nodes[0])});
    torch::Tensor w0 = torch::cat(
            {torch::full({1}, tab.centreWeight, kF64), blockWeights(tab.weights[0])});

    torch::Tensor fvals0 = pFunction(pos0);
    torch::Tensor integral = (w0.to(fvals0.options()) * fvals0).sum();

    // Target tolerance: identical to NumA (read from the owned integrator so the
    // default / any configured tolerance match bit-for-bit).
    double target = 1e-3;
    const double absTol = m_mathIntegrator->getTolerances().getAbsoluteTolerance();
    if (absTol < 2e-3) {
        target = absTol;
    }
    target /= c;

    double previousDelta = 0.;
    double currentDelta = std::numeric_limits<double>::max();
    double errorEstimate = std::numeric_limits<double>::max();
    double h = 1.0;

    for (int level = 1; level <= 9; ++level) {
        h *= 0.5;

        torch::Tensor pos = blockPositions(tab.nodes[level]);
        torch::Tensor w = blockWeights(tab.weights[level]);
        torch::Tensor fvals = pFunction(pos);
        torch::Tensor newContribution = (w.to(fvals.options()) * fvals).sum() * h;

        previousDelta = currentDelta;
        currentDelta = std::fabs(
                0.5 * integral.item<double>() - newContribution.item<double>());
        integral = 0.5 * integral + newContribution;

        if (level == 1) {
            continue; // previousDelta meaningless yet.
        }
        if (currentDelta == 0.0) {
            break;
        }

        const double r = std::log(currentDelta) / std::log(previousDelta);
        errorEstimate =
                (r > 1.9 && r < 2.1) ? currentDelta * currentDelta : currentDelta;

        if (errorEstimate < 0.1 * target) {
            break;
        }
    }

    return integral * c;
}
