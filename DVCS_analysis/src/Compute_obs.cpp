//
// Created by Mariana Khachatryan on 3/
#include "../include/Compute_obs.h"
#include <fstream>
#include <iomanip>

#include <ElementaryUtils/logger/LoggerManager.h>
#include <ElementaryUtils/parameters/Parameter.h>
#include <ElementaryUtils/parameters/Parameters.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <NumA/integration/one_dimension/QuadratureIntegrator1D.h>
#include <partons/beans/convol_coeff_function/ConvolCoeffFunctionResult.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/KinematicUtils.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/observable/ObservableResult.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/modules/active_flavors_thresholds/ActiveFlavorsThresholdsConstant.h>
#include <partons/modules/convol_coeff_function/ConvolCoeffFunctionModule.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSCFFStandard.h>
#include <partons/modules/evolution/gpd/GPDEvolutionVinnikov.h>
#include <partons/modules/gpd/GPDGK16.h>
#include <partons/modules/gpd/GPDGK16Numerical.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAllMinus.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluIntSin1Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinusSin1Phi.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluDVCSSin1Phi.h>
#include <partons/modules/process/DVCS/DVCSProcessGV08.h>
#include <partons/modules/running_alpha_strong/RunningAlphaStrongVinnikov.h>
#include <partons/modules/scales/DVCS/DVCSScalesQ2Multiplier.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterXBToXi.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/services/ConvolCoeffFunctionService.h>
#include <partons/services/DVCSConvolCoeffFunctionService.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/services/GPDService.h>
#include <partons/services/ObservableService.h>
#include <partons/ServiceObjectRegistry.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>




void ComputeManyKinematicsForDVCSObservable_BSA() {

    // Retrieve Observable service
    PARTONS::DVCSObservableService* pObservableService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();

    // Create GPDModule
    PARTONS::GPDModule* pGPDModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Create CFF module
    PARTONS::DVCSConvolCoeffFunctionModule* pDVCSCFFModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    PARTONS::DVCSCFFStandard::classId);

    // Set its PerturbativeQCDOrder
    pDVCSCFFModel->configure(
            ElemUtils::Parameter(
                    PARTONS::PerturbativeQCDOrderType::PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
                    PARTONS::PerturbativeQCDOrderType::LO));

    // Create XiConverterModule
    PARTONS::DVCSXiConverterModule* pXiConverterModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSXiConverterModule(
                    PARTONS::DVCSXiConverterXBToXi::classId);

    // Create ScalesModule
    PARTONS::DVCSScalesModule* pScalesModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSScalesModule(
                    PARTONS::DVCSScalesQ2Multiplier::classId);

    // Set its lambda parameter, so MuF2 = MuR2 = lambda * Q2
    pScalesModule->configure(
            ElemUtils::Parameter(
                    PARTONS::DVCSScalesQ2Multiplier::PARAMETER_NAME_LAMBDA,
                    1.));

    // Create ProcessModule
    PARTONS::DVCSProcessModule* pProcessModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSProcessModule(
                    PARTONS::DVCSProcessGV08::classId);

    // Create Observable
   /* PARTONS::DVCSObservable* pObservable =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    PARTONS::DVCSAllMinus::classId);//DVCS cross section for negative beam helicity, Observable (AllPlus / AllMinus / Asymmetry)
        PARTONS::DVCSObservable* pObservable = PARTONS::Partons::getInstance()->getModuleObjectFactory()
        ->newDVCSObservable(PARTONS::DVCSAluIntSin1Phi::classId);
        PARTONS::DVCSObservable* pObservable = PARTONS::Partons::getInstance()->getModuleObjectFactory()
        ->newDVCSObservable(PARTONS::DVCSAluDVCSSin1Phi::classId);
        */
        PARTONS::DVCSObservable* pObservable = PARTONS::Partons::getInstance()
      ->getModuleObjectFactory()
      ->newDVCSObservable(PARTONS::DVCSAluMinusSin1Phi::classId);

    // Link modules (set physics assumptions of your computation)
    pObservable->setProcessModule(pProcessModule);
    pProcessModule->setScaleModule(pScalesModule);
    pProcessModule->setXiConverterModule(pXiConverterModule);
    pProcessModule->setConvolCoeffFunctionModule(pDVCSCFFModel);
    pDVCSCFFModel->setGPDModule(pGPDModule);



    // Load list of kinematics from file
    PARTONS::List<PARTONS::DVCSObservableKinematic> observableKinematicList =
            PARTONS::KinematicUtils().getDVCSObservableKinematicFromFile(
                    "My_Analysis/Partons_input/BSA_CLAS_15_KK_format.csv");



    // Run computation
    PARTONS::List<PARTONS::DVCSObservableResult> observableResultList =
            pObservableService->computeManyKinematic(observableKinematicList,
                    pObservable);

    // Print results
    //PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,observableResultList.toString());
   //std::cout << observableKinematicList.toString().c_str() << std::endl;
   // std::cout << observableResultList.toString().c_str() << std::endl;
        std::ofstream out("My_Analysis/Partons_output/dvcs_DVCSAluSinPhi_BSACLAS15_Partons.csv");
        out << "xB,t,Q2,E,phi,DVCSAluSinPhi\n";

        for (size_t i = 0; i < observableKinematicList.size(); ++i) {

                const auto& kin = observableKinematicList[i];

                double xB   = kin.getXB().getValue();
                double Q2   = kin.getQ2().getValue();
                double t    = kin.getT().getValue();
                double E    = kin.getE().getValue();      // Beam energy
                double phi  = kin.getPhi().getValue();    // Angle in degrees

                double val  = observableResultList[i].getValue().getValue();

                out << std::fixed << std::setprecision(6)
                    << xB  << ","<< t  << ","<< Q2   << ","<< E   << ","<< phi << ","<< val << "\n";
        }

        out.close();


    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModule, 0);
    pGPDModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pDVCSCFFModel, 0);
    pDVCSCFFModel = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pXiConverterModule, 0);
    pXiConverterModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pScalesModule, 0);
    pScalesModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pProcessModule, 0);
    pProcessModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pObservable, 0);
    pObservable = 0;
}



