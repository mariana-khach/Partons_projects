//
// Created by Mariana Khachatryan on 3/25/26.
//

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/logger/LoggerManager.h>
#include <partons/Partons.h>
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

        CFF_NN_Fitter fitter(
            "/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_input/BSA_CLAS_07_KK_format_ALU_error.csv",
            0.3f,
            {"ImH"},// can be {"ReH", "ImH","ReE", "ImE","ReHt", "ImHt","ReEt", "ImEt"}
            0.0);  // x_pow: CFF = xB^x_pow * NNet_output; NN learns xB*CFF (~ O(1) near small xB)
        fitter.train_nn();
        fitter.predict();
        fitter.observ_calc();
        fitter.observ_calc_torch();
        fitter.observ_calc_torch_scalar();

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