/**
 * @file DVCSKinematicsTorch.cpp
 *
 * Implements Theory::computeKinematics().
 *
 * All formulae are taken directly from DVCSProcessBMJ12 (PARTONS),
 * specifically from:
 *   DVCSProcessModule::initModule()     → y, ε, tmin/tmax
 *   DVCSProcessBMJ12::initModule()      → K, K̃, cF, phaseSpace, P1/P2
 *   DVCSProcessBMJ12::computeFormFactors()
 *   DVCSProcessBMJ12::computeFourierCoeffsBH()
 *   DVCSProcessBMJ12::computeAngularCoeffsInterf()
 *
 * Only the unpolarised-target (Λ=0), electron-beam (charge –1) terms
 * are populated.  dC/dS arrays (LP target) are not computed.
 */

#include "../../../include/NNFit/theory/DVCSKinematicsTorch.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Theory {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static void computeFormFactors(DVCSKin& k)
{
    // Dipole parametrisation from BMJ12 (eq. matching PARTONS source)
    // Denominator factor
    const double denom  = std::pow(1. - 1.4084507042253522 * k.t, 2);
    const double t4M2m  = 4.*k.M2 - k.t;

    k.F1 = (4.*k.M2 - 2.79285*k.t) / (denom * t4M2m);
    k.F2 = (7.1714*k.M2)            / (denom * t4M2m);

    k.F1PlusF2      = k.F1 + k.F2;
    k.F1PlusF22     = k.F1PlusF2 * k.F1PlusF2;
    k.F1MinusDeltaF2 = k.F1*k.F1 - k.Delta2M2 * k.F2*k.F2;
    k.F1PlusDeltaF2  = k.F1      + k.Delta2M2 * k.F2;
}

static void computeBHCoefficients(DVCSKin& k)
{
    const double F1  = k.F1,  F2  = k.F2;
    const double F12  = F1*F1, F22 = F2*F2;
    const double F1F2 = k.F1PlusF22;           // (F1+F2)²
    const double FmD  = k.F1MinusDeltaF2;      // F1² – Δ²/(4M²) F2²
    const double dQ   = k.tQ2;                 // t/Q2
    const double dM   = k.Delta2M2;            // t/(4M²)
    const double t2   = k.t  * k.t;

    // ── c0 BH ──
    k.cBH[0] =
        8.*k.K2  * ( (2. + 3.*k.eps2)*FmD/dQ + 2.*k.xB2*F1F2 )
      + std::pow(2. - k.y, 2)
          * ( (2. + k.eps2)
                * ( k.xB2/dM * std::pow(1. + dQ, 2)
                  + 4.*(1. - k.xB)*(1. + k.xB*dQ) )
                * FmD
            + 4.*k.xB2
                * ( k.xB
                  + (1. - k.xB + k.eps2/2.) * std::pow(1. - dQ, 2)
                  - k.xB*(1. - 2.*k.xB)*t2/k.Q4 )
                * F1F2 )
      + 8.*(1. + k.eps2)
          * (1. - k.y - k.eps2*k.y2/4.)
          * ( 2.*k.eps2*(1. - dM)*FmD
            - k.xB2 * std::pow(1. - dQ, 2) * F1F2 );

    // ── c1 BH ──
    k.cBH[1] =
        8.*k.K*(2. - k.y)
          * ( (k.xB2/dM - 2.*k.xB - k.eps2)*FmD
            + 2.*k.xB2*(1. - (1. - 2.*k.xB)*dQ)*F1F2 );

    // ── c2 BH ──
    k.cBH[2] =
        8.*k.xB2*k.K2 * (FmD/dM + 2.*F1F2);
}

static void computeAngularCoeffs(DVCSKin& k)
{
    // Abbreviations following BMJ12 variable names
    const double tQ2   = k.tQ2;
    const double Kt2Q2 = k.Kt2Q2;
    const double tpQ2  = k.tpQ2;
    const double er0   = k.er[0];
    const double er1   = k.er[1];
    const double er2   = k.er[2];
    const double er3   = k.er[3];
    const double er4   = k.er[4];
    // er5 not needed for C/S arrays (only dC/dS)
    const double xB    = k.xB;
    const double xB2   = k.xB2;
    const double yv    = k.y;
    const double C2    = k.C2;
    const double B2    = k.B2;
    const double K0    = k.K;
    const double C1    = k.C1;
    const double eps2  = k.eps2;

    // helper: (2-y)^2
    const double d2y2  = (2.-yv)*(2.-yv);

    // ─── C++ block (i=0) ─────────────────────────────────────────────────
    // n=0
    k.C_ang[0][0][0] =
        -4.*(2.-yv)*(1.+er0)/er3
          * ( Kt2Q2*d2y2/er0
            + tQ2*C2*(2.-xB)
                * (1. + (2.*xB*(2.-xB+(er0-1.)/2.+eps2/(2.*xB))*tQ2
                         + eps2) / ((2.-xB)*(1.+er0)) ) );
    k.C_ang[0][1][0] =
        8.*(2.-yv)*xB*tQ2/er3
          * ( Kt2Q2*d2y2/er0
            + C2*(1.+er0)/2.*(1.+tQ2)
                * (1. + (er0-1.+2.*xB)*tQ2/(1.+er0)) );
    k.C_ang[0][2][0] =
        8.*(2.-yv)*tQ2/er3
          * ( Kt2Q2*d2y2/er0*(1.+er0-2.*xB)/2.
            - C2*(2.*Kt2Q2
                  - (1.+er0)/2.
                    * (1.+er0-xB
                       + (er0-1.+xB*(3.+er0-2.*xB)/(1.+er0))*tQ2 )) );

    // n=1
    k.C_ang[0][0][1] =
        -16.*K0*C2/er4
          * ( (1.+(1.-xB)*(er0-1.)/(2.*xB)+eps2/(4.*xB))*xB*tQ2 - 3.*eps2/4. )
        - 4.*K0*B2*(1.+er0-eps2)/er4
          * (1. - (1.-3.*xB)*tQ2
             + (1.-er0+3.*eps2)*xB*tQ2/(1.+er0-eps2));
    k.C_ang[0][1][1] =
        16.*K0*xB*tQ2/er4
          * ( d2y2*(1.-(1.-2.*xB)*tQ2)
            + C2*(1.+er0-2.*xB)*tpQ2/2. );
    k.C_ang[0][2][1] =
        -16.*K0*tQ2/er3
          * ( C2*(1.-(1.-2.*xB)*tQ2
                  + (4.*xB*(1.-xB)+eps2)*tpQ2/(4.*er0))
            - d2y2*(1.-xB/2.
                    + (1.+er0-2.*xB)/4.*(1.-tQ2)
                    + (4.*xB*(1.-xB)+eps2)*tpQ2/(2.*er0)) );

    // n=2
    k.C_ang[0][0][2] =
        8.*(2.-yv)*C2/er3
          * ( 2.*eps2*Kt2Q2/(er0+er1)
            + xB*tQ2*tpQ2
                * (1.-xB-(er0-1.)/2.+eps2/(2.*xB)) );
    k.C_ang[0][1][2] =
        8.*(2.-yv)*C2*xB*tQ2/er3
          * ( 4.*Kt2Q2/er0
            + (1.+er0-2.*xB)/2.*(1.+tQ2)*tpQ2 );
    k.C_ang[0][2][2] =
        4.*(2.-yv)*C2*tQ2/er3
          * ( 4.*(1.-2.*xB)*Kt2Q2/er0
            - (3.-er0-2.*xB+eps2/xB)*xB*tpQ2 );

    // n=3
    k.C_ang[0][0][3] =
        -8.*K0*C2*(er0-1.)/er4
          * ((1.-xB)*tQ2 + (er0-1.)/2.*(1.+tQ2));
    k.C_ang[0][1][3] =
        -8.*K0*C2*xB*tQ2/er4
          * (er0-1. + (1.+er0-2.*xB)*tQ2);
    k.C_ang[0][2][3] =
        16.*K0*C2*tQ2*tpQ2/er4
          * (xB*(1.-xB) + eps2/4.);

    // ── S++ block (i=0) ──────────────────────────────────────────────────
    // n=0 → zero (no s0)
    k.S_ang[0][0][0] = 0.;
    k.S_ang[0][1][0] = 0.;
    k.S_ang[0][2][0] = 0.;
    // n=1
    k.S_ang[0][0][1] =
        8.*K0*(2.-yv)*yv/er1
          * (1. + (1.-xB+(er0-1.)/2.)/er1*tpQ2);
    k.S_ang[0][1][1] =
        -8.*K0*(2.-yv)*yv*xB*tQ2/er3
          * (er0-1. + (1.+er0-2.*xB)*tQ2);
    k.S_ang[0][2][1] =
        8.*K0*(2.-yv)*yv*tQ2/er1
          * (1. - (1.-2.*xB)*(1.+er0-2.*xB)/(2.*er1)*tpQ2);
    // n=2
    k.S_ang[0][0][2] =
        -4.*C2*yv/er2
          * (1.+er0-2.*xB)*tpQ2
          * ( (eps2-xB*(er0-1.))/(1.+er0-2.*xB)
            - (2.*xB+eps2)*tpQ2/(2.*er0) );
    k.S_ang[0][1][2] =
        -4.*C2*yv*xB*tQ2/er3
          * (1.-(1.-2.*xB)*tQ2)
          * (er0-1.+(1.+er0-2.*xB)*tQ2);
    k.S_ang[0][2][2] =
        -8.*C2*yv*tQ2*tpQ2/er3
          * (1.-xB/2.+3.*eps2/4.)
          * (1.+er0-2.*xB)
          * (1. + (4.*(1.-xB)*xB+eps2)/(4.-2.*xB+3.*eps2)*tQ2);
    // n=3
    k.S_ang[0][0][3] = 0.;
    k.S_ang[0][1][3] = 0.;
    k.S_ang[0][2][3] = 0.;

    // ─── C0+ block (i=2) ─────────────────────────────────────────────────
    // n=0
    k.C_ang[2][0][0] =
        12.*std::sqrt(2.)*K0*(2.-yv)*C1/er4
          * (eps2 + (2.-6.*xB-eps2)/3.*tQ2);
    k.C_ang[2][1][0] =
        24.*std::sqrt(2.)*K0*(2.-yv)*C1*xB*tQ2/er4
          * (1.-(1.-2.*xB)*tQ2);
    k.C_ang[2][2][0] =
        4.*std::sqrt(2.)*K0*(2.-yv)*C1*tQ2/er4
          * (8.-6.*xB+5.*eps2)
          * (1. - tQ2*(2.-12.*xB*(1.-xB)-eps2)/(8.-6.*xB+5.*eps2));
    // n=1
    k.C_ang[2][0][1] =
        8.*std::sqrt(2.)*C1/er3
          * ( d2y2*tpQ2*(1.-xB + ((1.-xB)*xB + eps2/4.)/er0 * tpQ2)
            + C2/er0*(1.-(1.-2.*xB)*tQ2)
                * (eps2 - 2.*(1.+eps2/(2.*xB))*xB*tQ2) );
    k.C_ang[2][1][1] =
        16.*std::sqrt(2.)*C1*xB*tQ2/er4
          * (Kt2Q2*d2y2 + std::pow(1.-(1.-2.*xB)*tQ2,2)*C2);
    k.C_ang[2][2][1] =
        8.*std::sqrt(2.)*C1*tQ2/er4
          * ( Kt2Q2*(1.-2.*xB)*d2y2
            + (1.-(1.-2.*xB)*tQ2)*C2
                * (4.-2.*xB+3.*eps2+tQ2*(4.*xB*(1.-xB)+eps2)) );
    // n=2
    k.C_ang[2][0][2] =
        -8.*std::sqrt(2.)*K0*(2.-yv)*C1/er4
          * (1.+eps2/2.)
          * (1. + (1.+eps2/(2.*xB))/(1.+eps2/2.)*xB*tQ2);
    k.C_ang[2][1][2] =
        8.*std::sqrt(2.)*K0*(2.-yv)*C1*xB*tQ2/er4
          * (1.-(1.-2.*xB)*tQ2);
    k.C_ang[2][2][2] =
        8.*std::sqrt(2.)*K0*(2.-yv)*C1*tQ2/er3
          * (1.-xB + tpQ2/2.*(4.*xB*(1.-xB)+eps2)/er0);
    // n=3
    k.C_ang[2][0][3] = 0.;
    k.C_ang[2][1][3] = 0.;
    k.C_ang[2][2][3] = 0.;

    // ── S0+ block (i=2) ──────────────────────────────────────────────────
    k.S_ang[2][0][0] = 0.;
    k.S_ang[2][1][0] = 0.;
    k.S_ang[2][2][0] = 0.;
    // n=1
    k.S_ang[2][0][1] =
        8.*std::sqrt(2.)*(2.-yv)*yv*C1*Kt2Q2/er3;
    k.S_ang[2][1][1] =
        4.*std::sqrt(2.)*(2.-yv)*yv*C1*xB*tQ2/er3
          * ( 4.*(1.-xB)*tQ2*(1.+xB*tQ2)
            + eps2*(1.+tQ2)*(1.+tQ2) );
    k.S_ang[2][2][1] =
        -8.*std::sqrt(2.)*(2.-yv)*yv*(1.-2.*xB)*C1*tQ2*Kt2Q2/er3;
    // n=2
    k.S_ang[2][0][2] =
        8.*std::sqrt(2.)*K0*yv*C1*(1.+eps2/2.)/er3
          * (1. + (1.+eps2/(2.*xB))/(1.+eps2/2.)*xB*tQ2);
    k.S_ang[2][1][2] =
        -8.*std::sqrt(2.)*K0*yv*C1*xB*tQ2/er3
          * (1.-(1.-2.*xB)*tQ2);
    k.S_ang[2][2][2] =
        -2.*std::sqrt(2.)*K0*yv*C1*tQ2/er3
          * (4.-4.*xB+2.*eps2+2.*tQ2*(4.*xB*(1.-xB)+eps2));
    k.S_ang[2][0][3] = 0.;
    k.S_ang[2][1][3] = 0.;
    k.S_ang[2][2][3] = 0.;

    // ─── C-+ block (i=1) ─────────────────────────────────────────────────
    // n=0
    k.C_ang[1][0][0] =
        8.*(2.-yv)/er2
          * ( d2y2*(er0-1.)/(2.*er1)*Kt2Q2
            + C2/er0*(1.-xB-(er0-1.)/2.+eps2/(2.*xB))*xB*tQ2*tpQ2 );
    k.C_ang[1][1][0] =
        4.*(2.-yv)*xB*tQ2/er4
          * ( 2.*Kt2Q2*B2
            - (1.-(1.-2.*xB)*tQ2)*C2
                * (er0-1.+(er0+1.-2.*xB)*tQ2) );
    k.C_ang[1][2][0] =
        4.*(2.-yv)*tQ2/er3
          * ( tpQ2*C2
                * (2.*xB2-eps2-3.*xB+xB*er0)
            + Kt2Q2/er0
                * (4.-2.*xB*(2.-yv)*(2.-yv)-4.*yv+yv*yv-yv*yv*er2) );
    // n=1
    k.C_ang[1][0][1] =
        8.*K0/er2
          * ( d2y2*(2.-er0)/er1
                * ((er0-1.+eps2)/(2.*(2.-er0))*(1.-tQ2) - xB*tQ2)
            + 2.*C2/er0
                * ((1.-er0+eps2/2.)/(2.*er0)
                   + tQ2*(1.-3.*xB/2.+(xB+eps2/2.)/(2.*er0))) );
    k.C_ang[1][1][1] =
        8.*K0*xB*tQ2/er4
          * ( 2.*(1.-(1.-2.*xB)*tQ2)*B2
            + C2*(3.-er0-(3.*(1.-2.*xB)+er0)*tQ2) );
    k.C_ang[1][2][1] =
        4.*K0*tQ2/er4
          * ( B2*(5.-4.*xB+3.*eps2-er0
                  - tQ2*(1.-eps2-er0-2.*xB*(4.-4.*xB-er0)))
            + C2*(8.+5.*eps2-6.*xB+2.*xB*er0
                  - tQ2*(2.-eps2+2.*er0-4.*xB*(3.-3.*xB+er0))) );
    // n=2
    k.C_ang[1][0][2] =
        4.*(2.-yv)*C2*(1.+er0)/er4
          * ( (2.-3.*xB)*tQ2
            + (1.-2.*xB+2.*(1.-xB)/(1.+er0))*xB*tQ2*tQ2
            + (1.+(er0+xB+(1.-xB)*tQ2)/(1.+er0)*tQ2)*eps2 );
    k.C_ang[1][1][2] =
        4.*(2.-yv)*C2*xB*tQ2/er4
          * ( 4.*Kt2Q2+1.+er0
            + tQ2*((1.-2.*xB)*(1.-2.*xB-er0)*tQ2
                   -2.+4.*xB+2.*xB*er0) );
    k.C_ang[1][2][2] =
        16.*(2.-yv)*C2*tQ2/er2
          * ( Kt2Q2*(1.-2.*xB)/er1
            - (1.-xB)/(4.*xB*(1.-xB)+eps2)
                * (2.*xB2-eps2-3.*xB-xB*er0)
            - tpQ2*(2.*xB2-eps2-3.*xB-xB*er0)/(4.*er0) );
    // n=3
    k.C_ang[1][0][3] =
        -8.*K0*C2*(1.+er0+eps2/2.)/er4
          * (1. + (1.+er0+eps2/(2.*xB))/(1.+er0+eps2/2.)*xB*tQ2);
    k.C_ang[1][1][3] =
        8.*K0*C2*xB*tQ2/er4*(1.+er0)
          * (1. - tQ2*(1.-2.*xB-er0)/(1.+er0));
    k.C_ang[1][2][3] =
        16.*K0*C2*tQ2/er3
          * (1.-xB + tpQ2*(4.*xB*(1.-xB)+eps2)/(4.*er0));

    // ── S-+ block (i=1) ──────────────────────────────────────────────────
    k.S_ang[1][0][0] = 0.;
    k.S_ang[1][1][0] = 0.;
    k.S_ang[1][2][0] = 0.;
    // n=1
    k.S_ang[1][0][1] =
        4.*K0*(2.-yv)*yv/er3
          * (1.-er0+2.*eps2-2.*(1.+(er0-1.)/(2.*xB))*xB*tQ2);
    k.S_ang[1][1][1] =
        8.*K0*(2.-yv)*yv*xB*tQ2/er3*(1.+er0)
          * (1.-tQ2*(1.-2.*xB-er0)/(1.+er0));
    k.S_ang[1][2][1] =
        4.*K0*(2.-yv)*yv*tQ2/er3
          * (3.+2.*eps2+er0-2.*xB*(1.+er0)
             - tQ2*(1.-2.*xB)*(1.-2.*xB-er0));
    // n=2
    k.S_ang[1][0][2] =
        2.*yv*C2*(1.+er0)/er3
          * (eps2-2.*(1.+eps2/(2.*xB))*xB*tQ2)
          * (1.+(er0-1.+2.*xB)/(1.+er0)*tQ2);
    k.S_ang[1][1][2] =
        4.*yv*C2*xB*tQ2/er3*(1.+er0)
          * (1.-(1.-2.*xB)*tQ2)
          * (1.-tQ2*(1.-2.*xB-er0)/(1.+er0));
    k.S_ang[1][2][2] =
        2.*yv*C2*tQ2/er3
          * (4.-2.*xB+3.*eps2+tQ2*(4.*xB*(1.-xB)+eps2))
          * (1.+er0-tQ2*(1.-2.*xB-er0));
    k.S_ang[1][0][3] = 0.;
    k.S_ang[1][1][3] = 0.;
    k.S_ang[1][2][3] = 0.;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public factory
// ─────────────────────────────────────────────────────────────────────────────

DVCSKin computeKinematics(double xB, double t, double Q2, double E)
{
    DVCSKin k{};

    // ── Primary inputs ──────────────────────────────────────────────────
    k.xB  = xB;
    k.t   = t;
    k.Q2  = Q2;
    k.E   = E;

    // ── Proton mass ─────────────────────────────────────────────────────
    k.M  = M_PROTON;
    k.M2 = k.M * k.M;

    // ── Photon virtuality ───────────────────────────────────────────────
    k.Q  = std::sqrt(Q2);
    k.Q3 = k.Q  * Q2;
    k.Q4 = Q2 * Q2;

    // ── Bjorken x ───────────────────────────────────────────────────────
    k.xB2 = xB * xB;

    // ── Inelasticity y (Eq. DVCSProcessModule::initModule) ──────────────
    k.y  = Q2 / (2.*xB*k.M*E);
    k.y2 = k.y * k.y;
    k.y3 = k.y * k.y2;

    // ── Epsilon ──────────────────────────────────────────────────────────
    k.eps  = 2.*xB*k.M / k.Q;
    k.eps2 = k.eps * k.eps;

    // ── epsroot[k] array (PARTONS convention) ────────────────────────────
    k.er[1] = 1. + k.eps2;                  // (1+ε²)
    k.er[0] = std::sqrt(k.er[1]);           // √(1+ε²)
    k.er[2] = k.er[1] * k.er[0];           // (1+ε²)^{3/2}
    k.er[3] = k.er[1] * k.er[1];           // (1+ε²)²
    k.er[4] = k.er[2] * k.er[1];           // (1+ε²)^{5/2}
    k.er[5] = k.er[2] * k.er[2];           // (1+ε²)^3

    // ── Composite kinematic factor xBtQ2 ─────────────────────────────────
    k.xBtQ2_0 = 2. - xB + xB*t/Q2;
    k.xBtQ2_1 = k.xBtQ2_0 * k.xBtQ2_0;
    k.xBtQ2_2 = k.xBtQ2_1 * k.xBtQ2_0;

    // ── t/Q2 and t/(4M²) ─────────────────────────────────────────────────
    k.tQ2      = t / Q2;
    k.Delta2M2 = t / (4.*k.M2);

    // ── Kinematic limits (DVCSProcessModule::initModule) ─────────────────
    {
        const double tfactor = -Q2 / (4.*xB*(1.-xB) + k.eps2);
        k.tmin = tfactor * (2.*(1.-xB)*(1.-k.er[0]) + k.eps2);
        k.tmax = tfactor * (2.*(1.-xB)*(1.+k.er[0]) + k.eps2);
    }
    k.tp = t - k.tmin;

    // ── K² and K̃² (DVCSProcessBMJ12::initModule) ──────────────────────
    k.K2 = -(t/Q2)*(1.-xB)
           * (1. - k.y - k.y2*k.eps2/4.)
           * (1. - k.tmin/t)
           * (k.er[0]
              + (4.*xB*(1.-xB)+k.eps2)/(4.*(1.-xB))*(t - k.tmin)/Q2);
    k.K  = std::sqrt(std::max(k.K2, 0.));

    k.Kt2 = ((1.-xB)*xB + k.eps2/4.) * (k.tmin - t)*(t - k.tmax) / Q2;
    k.Kt  = std::sqrt(std::max(k.Kt2, 0.));

    // ── Derived t-ratios ─────────────────────────────────────────────────
    k.Kt2Q2 = k.Kt2 / Q2;
    k.tpQ2  = k.tp  / Q2;

    // ── Form factors ─────────────────────────────────────────────────────
    computeFormFactors(k);

    // ── Dressed-CFF scaling coefficients ─────────────────────────────────
    // j=0 (+,+), j=1 (+,–)
    for (int j = 0; j < 2; j++) {
        const double sign_j = (j == 0) ? 1. : -1.;
        k.cF[j] = (1. + sign_j*k.er[0]) / (2.*k.er[0])
                + (1.-xB)*k.xB2*(4.*k.M2 - t)*(1. + t/Q2)
                  / (Q2*k.er[0]*k.xBtQ2_1);
    }
    // j=2 (0,+)
    {
        const double A = -std::sqrt(2.)*k.Kt / (k.er[0]*k.Q*k.xBtQ2_0);
        k.cF[2] = A * xB
                * (1. + 2.*xB*(4.*k.M2 - t)/(Q2*k.xBtQ2_0));
    }

    // ── Lepton propagator decomposition ──────────────────────────────────
    // phi1 = π – φ  →  cos(phi1) = –cos(φ).
    // P1(φ) = –(J + 2K cos(phi1)) / yeps
    //        = –J/yeps + 2K cos(φ) / yeps   [because cos(phi1) = –cos(φ)]
    // We store P1_c (constant) and P1_K (coefficient of cos(φ)).
    {
        const double dQ  = k.tQ2;
        k.yeps   = k.y * k.er[1];
        k.J_val  = (1. - k.y - k.y*k.eps2/2.)*(1. + dQ)
                 - (1. - xB)*(2. - k.y)*dQ;
        k.P1_c   = -k.J_val / k.yeps;
        k.P1_K   =  2.*k.K  / k.yeps;   // +coefficient of cos(φ)
        k.P12_sum = 1. + dQ;             // P1 + P2
    }

    // ── Phase-space factor ────────────────────────────────────────────────
    k.phaseSpace = xB * k.y2 / (1024.*std::pow(PI, 5)*k.Q4*k.er[0]);

    // ── y-ε combinations ─────────────────────────────────────────────────
    k.C2 = 1. - k.y - k.eps2*k.y2/4.;
    k.C1 = std::sqrt(std::max(k.C2, 0.));
    k.B2 = 2. - 2.*k.y + k.y2 + k.eps2*k.y2/2.;

    // ── BH Fourier coefficients ──────────────────────────────────────────
    computeBHCoefficients(k);

    // ── Angular coefficients for the interference term ────────────────────
    computeAngularCoeffs(k);

    return k;
}

} // namespace Theory