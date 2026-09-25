//
// Created by Mariana Khachatryan on 3/25/26.
//

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/logger/LoggerManager.h>
#include <partons/Partons.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUMinus.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionDifferenceLUMinus.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUBHSubProc.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUDVCSSubProc.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUVirtualPhotoProduction.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUMinusPhiIntegrated.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUDVCSSubProcPhiIntegrated.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAc.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAcCos0Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAcCos1Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAcCos2Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAcCos3Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinus.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluPlus.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluDVCS.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluInt.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinusSin2Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluDVCSSin1Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluIntSin1Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluIntSin2Phi.h>
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUMinusTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionDifferenceLUMinusTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUBHSubProcTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUDVCSSubProcTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUVirtualPhotoProductionTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUMinusPhiIntegratedTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAcTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAcCos0PhiTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAcCos1PhiTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAcCos2PhiTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAcCos3PhiTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluPlusTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluDVCSTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluIntTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusSin2PhiTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluDVCSSin1PhiTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluIntSin1PhiTorch.h"
#include "../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluIntSin2PhiTorch.h"
#include "../include/NNFit/CFF_NN_Fit.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {

    PARTONS::Partons* pPartons = 0;
    const auto start_time = std::chrono::steady_clock::now();

    try {

        pPartons = PARTONS::Partons::getInstance();
        pPartons->init(argc, argv);

        // Fitting the beam-spin DIFFERENCE cross section, d^4(sigma+ - sigma-)/2,
        // CLAS 2018 (3008 points). The observable is taken from the data file's
        // header (DVCSCrossSectionDifferenceLUMinus -> the tensor leaf
        // ...Torch), so only the path and the output layer are set here.
        //
        // ImH, not ReH: this is a helicity-ODD observable, so it isolates the
        // interference term's SINE harmonics, which carry the IMAGINARY parts
        // of the CFFs -- the same pairing as A_LU, and the opposite of A_C.
        //
        // NOTE this is the first POINTWISE (per-phi) dataset: phi varies row to
        // row (230 distinct values over 1250 kinematic bins), so the leaf
        // evaluates each point at its OWN phi rather than integrating phi away.
        // It is also ~170x larger than the previous sets -- see the per-epoch
        // cost before launching a full replica ensemble.
        CFF_NN_Fitter fitter(
            "/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_input/BSD_CLAS_18_MH_format_XLU_phi_error.csv",
            0.3f,
            {"ImH"},// can be {"ReH", "ImH","ReE", "ImE","ReHt", "ImHt","ReEt", "ImEt"}
            0.0);  // x_pow: CFF = xB^x_pow * NNet_output; NN learns xB*CFF (~ O(1) near small xB)
        fitter.train_nn();
        fitter.predict();
        fitter.observ_calc();
        fitter.observ_calc_torch();
        fitter.observ_calc_torch_scalar();

        // ---- BMJ12 differential tests (OFF) --------------------------------
        // 28 checks that push identical fixed CFFs (DVCSCFFConstant) through
        // PARTONS' native scalar chain and through the tensor one, so any
        // disagreement is in the BMJ12 transcription and nothing else. They
        // have served their purpose: all 22 ported observables pass, at CLAS
        // and at HERMES kinematics.
        //
        // Off for production fits because the NATIVE side loops per data point
        // (PARTONS has no batched entry point that keeps per-point values), so
        // the cost scales with the data file: ~4 s on a 16-point file, but
        // 28 x 3008 ~ 84000 scalar evaluations -- most with adaptive DEXP
        // phi-integration -- on the CLAS 2018 per-phi set.
        //
        // Turn back on after ANY change to DVCSProcessBMJ12Torch, to a leaf's
        // formula, or when adding an observable. Prefer a small data file when
        // doing so: 16 well-spread points already exercise the transcription.
        const bool runBMJ12DifferentialTests = false;
        if (runBMJ12DifferentialTests) {
            // Network-free differential test of the batched BMJ12 port: fixed CFFs
            // through PARTONS' native process module vs the tensor one.
            // Every A_LU observable, checked against the PARTONS class it mirrors:
            // identical fixed CFFs through the native chain and the tensor one.
            fitter.observ_calc_scalar_cff();   // DVCSAluMinusSin1Phi (the default)
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluMinus::classId,
                    DVCSAluMinusTorch::classId, "DVCSAluMinus (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluPlus::classId,
                    DVCSAluPlusTorch::classId, "DVCSAluPlus (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluDVCS::classId,
                    DVCSAluDVCSTorch::classId, "DVCSAluDVCS (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluInt::classId,
                    DVCSAluIntTorch::classId, "DVCSAluInt (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluMinusSin2Phi::classId,
                    DVCSAluMinusSin2PhiTorch::classId, "DVCSAluMinusSin2Phi (moment)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluDVCSSin1Phi::classId,
                    DVCSAluDVCSSin1PhiTorch::classId, "DVCSAluDVCSSin1Phi (moment)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluIntSin1Phi::classId,
                    DVCSAluIntSin1PhiTorch::classId, "DVCSAluIntSin1Phi (moment)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAluIntSin2Phi::classId,
                    DVCSAluIntSin2PhiTorch::classId, "DVCSAluIntSin2Phi (moment)");

            // Beam-charge asymmetry family. The pointwise leaf is checked twice:
            // once at the data file's phi and once with phi swept over [0, 2pi),
            // because every row of that file carries the same phi and a charge
            // combination that is wrong elsewhere in phi would otherwise pass.
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAc::classId,
                    DVCSAcTorch::classId, "DVCSAc (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAc::classId,
                    DVCSAcTorch::classId, "DVCSAc (pointwise, phi swept)", true);
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAcCos0Phi::classId,
                    DVCSAcCos0PhiTorch::classId, "DVCSAcCos0Phi (average, /2pi)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAcCos1Phi::classId,
                    DVCSAcCos1PhiTorch::classId, "DVCSAcCos1Phi (moment)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAcCos2Phi::classId,
                    DVCSAcCos2PhiTorch::classId, "DVCSAcCos2Phi (moment)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSAcCos3Phi::classId,
                    DVCSAcCos3PhiTorch::classId, "DVCSAcCos3Phi (moment)");

            // Cross sections -- dimensionful (nb), unlike every leaf above.
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUMinus::classId,
                    DVCSCrossSectionUUMinusTorch::classId, "DVCSCrossSectionUUMinus (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUMinus::classId,
                    DVCSCrossSectionUUMinusTorch::classId, "DVCSCrossSectionUUMinus (phi swept)", true);
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionDifferenceLUMinus::classId,
                    DVCSCrossSectionDifferenceLUMinusTorch::classId, "DVCSCrossSectionDifferenceLUMinus (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionDifferenceLUMinus::classId,
                    DVCSCrossSectionDifferenceLUMinusTorch::classId, "DVCSCrossSectionDifferenceLUMinus (phi swept)", true);
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUBHSubProc::classId,
                    DVCSCrossSectionUUBHSubProcTorch::classId, "DVCSCrossSectionUUBHSubProc (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUBHSubProc::classId,
                    DVCSCrossSectionUUBHSubProcTorch::classId, "DVCSCrossSectionUUBHSubProc (phi swept)", true);
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUDVCSSubProc::classId,
                    DVCSCrossSectionUUDVCSSubProcTorch::classId, "DVCSCrossSectionUUDVCSSubProc (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUDVCSSubProc::classId,
                    DVCSCrossSectionUUDVCSSubProcTorch::classId, "DVCSCrossSectionUUDVCSSubProc (phi swept)", true);
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUVirtualPhotoProduction::classId,
                    DVCSCrossSectionUUVirtualPhotoProductionTorch::classId, "DVCSCrossSectionUUVirtualPhotoProduction (pointwise)");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUVirtualPhotoProduction::classId,
                    DVCSCrossSectionUUVirtualPhotoProductionTorch::classId, "DVCSCrossSectionUUVirtualPhotoProduction (phi swept)", true);
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUMinusPhiIntegrated::classId,
                    DVCSCrossSectionUUMinusPhiIntegratedTorch::classId, "DVCSCrossSectionUUMinusPhiIntegrated");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUDVCSSubProcPhiIntegrated::classId,
                    DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch::classId, "DVCSCrossSectionUUDVCSSubProcPhiIntegrated");
            fitter.observ_calc_scalar_cff(PARTONS::DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated::classId,
                    DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch::classId, "DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated");
        }

        // Monte Carlo replica ensemble (smeared pseudodata) for a CFF
        // uncertainty band, alongside the central fit above.
        fitter.train_replicas(10);
        fitter.export_replicas(CFF_NN_Fitter::OUT_DIR);

    } catch (const ElemUtils::CustomException &e) {
        pPartons->getLoggerManager()->error(e);
        if (pPartons) pPartons->close();
    } catch (const std::exception &e) {
        pPartons->getLoggerManager()->error("main", __func__, e.what());
        if (pPartons) pPartons->close();
    }

    if (pPartons) pPartons->close();

    const auto end_time = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total run time: " << elapsed_s << " s\n";

    return 0;
}