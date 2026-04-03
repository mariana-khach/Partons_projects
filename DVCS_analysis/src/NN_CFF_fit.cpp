//
// Created by Mariana Khachatryan on 3/25/26.
//

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/logger/LoggerManager.h>
#include <partons/Partons.h>
#include "../include/NNFit/NN_Fit.h"

int main(int argc, char** argv) {

    PARTONS::Partons* pPartons = 0;

    try {

        pPartons = PARTONS::Partons::getInstance();
        pPartons->init(argc, argv);

        NN_Fitter fitter(
            "/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_input/BSA_CLAS_07_KK_format_ALU.csv",
            0.25f,
            {"ImH"});
        fitter.train_nn();
        fitter.predict();

    } catch (const ElemUtils::CustomException &e) {
        pPartons->getLoggerManager()->error(e);
        if (pPartons) pPartons->close();
    } catch (const std::exception &e) {
        pPartons->getLoggerManager()->error("main", __func__, e.what());
        if (pPartons) pPartons->close();
    }

    if (pPartons) pPartons->close();

    return 0;
}