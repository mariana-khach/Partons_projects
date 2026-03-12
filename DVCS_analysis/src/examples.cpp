#include "../include/examples.h"

#include <ElementaryUtils/logger/LoggerManager.h>
#include <ElementaryUtils/parameters/Parameter.h>
#include <ElementaryUtils/parameters/Parameters.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <NumA/integration/one_dimension/QuadratureIntegrator1D.h>
#include <partons/beans/convol_coeff_function/ConvolCoeffFunctionResult.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/convol_coeff_function/DVMP/DVMPConvolCoeffFunctionKinematic.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/KinematicUtils.h>
#include <partons/beans/List.h>
#include <partons/beans/PolarizationType.h>
#include <partons/beans/MesonType.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/observable/DVMP/DVMPObservableKinematic.h>
#include <partons/beans/observable/ObservableResult.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/modules/active_flavors_thresholds/ActiveFlavorsThresholdsConstant.h>
#include <partons/modules/active_flavors_thresholds/ActiveFlavorsThresholdsQuarkMasses.h>
#include <partons/modules/convol_coeff_function/ConvolCoeffFunctionModule.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSCFFStandard.h>
#include <partons/modules/convol_coeff_function/DVMP/DVMPCFFGK06.h>
#include <partons/modules/evolution/gpd/GPDEvolutionVinnikov.h>
#include <partons/modules/gpd/GPDGK16Numerical.h>
#include <partons/modules/gpd/GPDGK19.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAllMinus.h>
#include <partons/modules/observable/DVMP/cross_section/DVMPCrossSectionUUUMinus.h>
#include <partons/modules/process/DVCS/DVCSProcessGV08.h>
#include <partons/modules/process/DVMP/DVMPProcessGK06.h>
#include <partons/modules/running_alpha_strong/RunningAlphaStrongVinnikov.h>
#include <partons/modules/scales/DVCS/DVCSScalesQ2Multiplier.h>
#include <partons/modules/scales/DVMP/DVMPScalesQ2Multiplier.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterXBToXi.h>
#include <partons/modules/xi_converter/DVMP/DVMPXiConverterXBToXi.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/services/ConvolCoeffFunctionService.h>
#include <partons/services/DVCSConvolCoeffFunctionService.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/services/DVMPConvolCoeffFunctionService.h>
#include <partons/services/DVMPObservableService.h>
#include <partons/services/GPDService.h>
#include <partons/services/ObservableService.h>
#include <partons/ServiceObjectRegistry.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <partons/modules/collinear_distribution/CollinearDistributionLHAPDF.h>
#include <partons/services/CollinearDistributionService.h>

void computeSingleKinematicsForCollinearDistribution() {

    // Retrieve CollinearDistribution service
    PARTONS::CollinearDistributionService* pCollinearDistributionService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getCollinearDistributionService();

    // Create CollinearDistribution module with the BaseModuleFactory
    PARTONS::CollinearDistributionModule* pCollinearDistributionModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newCollinearDistributionModule(
                    PARTONS::CollinearDistributionLHAPDF::classId);

    // Create parameters to configure later GPDEvolutionModule
    ElemUtils::Parameters parameters;

    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::CollinearDistributionType::COLLINEAR_DISTRIBUTION_TYPE_DB_COLUMN_NAME,
                    PARTONS::CollinearDistributionType::Type::UnpolPDF));

    parameters.add(
            ElemUtils::Parameter(
		    PARTONS::CollinearDistributionLHAPDF::PARAM_NAME_SET_NAME,
                    "CT14nnlo"));

    parameters.add(
            ElemUtils::Parameter(
		    PARTONS::CollinearDistributionLHAPDF::PARAM_NAME_SET_MEMBER,
                    0));

    // Configure collinear-distribution module with previous parameters.
    pCollinearDistributionModel->configure(parameters);

    // Create a CollinearDistributionKinematic(x, MuF2, MuR2) to compute
    PARTONS::CollinearDistributionKinematic colldistKinematic(0.1, 2., 2.);

    // Run computation
    PARTONS::CollinearDistributionResult colldistResult =
            pCollinearDistributionService->computeSingleKinematic(
                    colldistKinematic, pCollinearDistributionModel);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            colldistResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pCollinearDistributionModel, 0);
    pCollinearDistributionModel = 0;
}

void computeSingleKinematicsForGPD() {

    // Retrieve GPD service
    PARTONS::GPDService* pGPDService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getGPDService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Create a GPDKinematic(x, xi, t, MuF2, MuR2) to compute
    PARTONS::GPDKinematic gpdKinematic(0.1, 0.2, -0.1, 2., 2.);

    // Run computation
    PARTONS::GPDResult gpdResult = pGPDService->computeSingleKinematic(
            gpdKinematic, pGPDModel);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            gpdResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModel, 0);
    pGPDModel = 0;
}

void computeManyKinematicsForGPD() {

    // Retrieve GPD service
    PARTONS::GPDService* pGPDService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getGPDService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Load list of kinematics from file
    PARTONS::List<PARTONS::GPDKinematic> gpdKinematicList =
            PARTONS::KinematicUtils().getGPDKinematicFromFile(
                    "/home/partons/git/partons-example/data/examples/gpd/kinematics_gpd.csv");

    // Run computation
    PARTONS::List<PARTONS::GPDResult> gpdResultList =
            pGPDService->computeManyKinematic(gpdKinematicList, pGPDModel);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            gpdResultList.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModel, 0);
    pGPDModel = 0;
}

void computeSingleKinematicsForDVCSComptonFormFactor() {

    // Retrieve service
    PARTONS::DVCSConvolCoeffFunctionService* pDVCSConvolCoeffFunctionService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getDVCSConvolCoeffFunctionService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Create CFF module with the BaseModuleFactory
    PARTONS::DVCSConvolCoeffFunctionModule* pDVCSCFFModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    PARTONS::DVCSCFFStandard::classId);

    // Create parameters to configure later DVCSCFFModel with PerturbativeQCD = LO
    ElemUtils::Parameters parameters(
            PARTONS::PerturbativeQCDOrderType::PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO);

    // Configure DVCSCFFModule with previous parameters.
    pDVCSCFFModule->configure(parameters);

    // Link modules (set physics assumptions of your computation)
    pDVCSCFFModule->setGPDModule(pGPDModule);

    // Create kinematic
    PARTONS::DVCSConvolCoeffFunctionKinematic cffKinematic =
            PARTONS::DVCSConvolCoeffFunctionKinematic(0.01, -0.1, 4., 4., 4.);

    // Run computation
    PARTONS::DVCSConvolCoeffFunctionResult cffResult =
            pDVCSConvolCoeffFunctionService->computeSingleKinematic(
                    cffKinematic, pDVCSCFFModule);

    // Print results for DVCSCFFModule
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            cffResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pDVCSCFFModule, 0);
    pDVCSCFFModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModule, 0);
    pGPDModule = 0;
}

void computeManyKinematicsForDVCSComptonFormFactor() {

    // Retrieve service
    PARTONS::DVCSConvolCoeffFunctionService* pDVCSConvolCoeffFunctionService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getDVCSConvolCoeffFunctionService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Create CFF module with the BaseModuleFactory
    PARTONS::DVCSConvolCoeffFunctionModule* pDVCSCFFModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    PARTONS::DVCSCFFStandard::classId);

    // Create parameters to configure later DVCSCFFModel with PerturbativeQCD = LO
    ElemUtils::Parameters parameters(
            PARTONS::PerturbativeQCDOrderType::PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO);

    // Configure DVCSCFFModule with previous parameters.
    pDVCSCFFModule->configure(parameters);

    // Link modules (set physics assumptions of your computation)
    pDVCSCFFModule->setGPDModule(pGPDModule);

    // Load list of kinematics from file
    PARTONS::List<PARTONS::DVCSConvolCoeffFunctionKinematic> cffKinematicList =
            PARTONS::KinematicUtils().getDVCSCCFKinematicFromFile(
                    "/home/partons/git/partons-example/data/examples/cff/kinematics_dvcs_cff.csv");

    // Run computation
    PARTONS::List<PARTONS::DVCSConvolCoeffFunctionResult> cffResultList =
            pDVCSConvolCoeffFunctionService->computeManyKinematic(
                    cffKinematicList, pDVCSCFFModule);

    // Print results for DVCSCFFModule
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            cffResultList.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pDVCSCFFModule, 0);
    pDVCSCFFModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModule, 0);
    pGPDModule = 0;
}

void computeSingleKinematicsForDVMPComptonFormFactor() {

    // Retrieve service
    PARTONS::DVMPConvolCoeffFunctionService* pDVMPConvolCoeffFunctionService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getDVMPConvolCoeffFunctionService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK19::classId);

    // Create CFF module with the BaseModuleFactory
    PARTONS::DVMPConvolCoeffFunctionModule* pDVMPCFFModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVMPConvolCoeffFunctionModule(
                    PARTONS::DVMPCFFGK06::classId);

    // Create parameters to configure later DVMPCFFModel with PerturbativeQCD = LO
    ElemUtils::Parameters parameters;

    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::PerturbativeQCDOrderType::PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
                    PARTONS::PerturbativeQCDOrderType::LO));

    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::DVMPCFFGK06::PARAMETER_NAME_DVMPCFFGK06_MC_NWARMUP,
                    10000));
    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::DVMPCFFGK06::PARAMETER_NAME_DVMPCFFGK06_MC_NCALLS,
                    100000));
    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::DVMPCFFGK06::PARAMETER_NAME_DVMPCFFGK06_MC_CHI2LIMIT,
                    0.8));

    // Configure DVMPCFFModule with previous parameters.
    pDVMPCFFModule->configure(parameters);

    // Link modules (set physics assumptions of your computation)
    pDVMPCFFModule->setGPDModule(pGPDModule);

    // Create kinematic
    PARTONS::DVMPConvolCoeffFunctionKinematic cffKinematic =
            PARTONS::DVMPConvolCoeffFunctionKinematic(0.01, -0.1, 4., 4., 4.,
                    PARTONS::MesonType::PI0,
                    PARTONS::PolarizationType::UNDEFINED);

    // Run computation
    PARTONS::DVMPConvolCoeffFunctionResult cffResult =
            pDVMPConvolCoeffFunctionService->computeSingleKinematic(
                    cffKinematic, pDVMPCFFModule);

    // Print results for DVMPCFFModule
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            cffResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pDVMPCFFModule, 0);
    pDVMPCFFModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModule, 0);
    pGPDModule = 0;
}

void computeSingleKinematicsForDVCSObservable() {

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
    PARTONS::DVCSObservable* pObservable =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    PARTONS::DVCSAllMinus::classId);

    // Link modules (set physics assumptions of your computation)
    pObservable->setProcessModule(pProcessModule);
    pProcessModule->setScaleModule(pScalesModule);
    pProcessModule->setXiConverterModule(pXiConverterModule);
    pProcessModule->setConvolCoeffFunctionModule(pDVCSCFFModel);
    pDVCSCFFModel->setGPDModule(pGPDModule);

    // Load list of kinematics from file
    PARTONS::DVCSObservableKinematic observableKinematic =
            PARTONS::DVCSObservableKinematic(0.2, -0.1, 2., 6., 0.);

    // Create kinematic
    PARTONS::DVCSObservableResult observableResult =
            pObservableService->computeSingleKinematic(observableKinematic,
                    pObservable);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            observableResult.toString());

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

void computeManyKinematicsForDVCSObservable() {

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
    PARTONS::DVCSObservable* pObservable =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    PARTONS::DVCSAllMinus::classId);

    // Link modules (set physics assumptions of your computation)
    pObservable->setProcessModule(pProcessModule);
    pProcessModule->setScaleModule(pScalesModule);
    pProcessModule->setXiConverterModule(pXiConverterModule);
    pProcessModule->setConvolCoeffFunctionModule(pDVCSCFFModel);
    pDVCSCFFModel->setGPDModule(pGPDModule);

    // Load list of kinematics from file
    PARTONS::List<PARTONS::DVCSObservableKinematic> observableKinematicList =
            PARTONS::KinematicUtils().getDVCSObservableKinematicFromFile(
                    "/home/partons/git/partons-example/data/examples/observable/kinematics_dvcs_observable.csv");

    // Run computation
    PARTONS::List<PARTONS::DVCSObservableResult> observableResultList =
            pObservableService->computeManyKinematic(observableKinematicList,
                    pObservable);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            observableResultList.toString());

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

void computeSingleKinematicsForDVMPObservable() {

    // Retrieve Observable service
    PARTONS::DVMPObservableService* pObservableService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getDVMPObservableService();

    // Create GPDModule
    PARTONS::GPDModule* pGPDModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK19::classId);

    // Create CFF module with the BaseModuleFactory
    PARTONS::DVMPConvolCoeffFunctionModule* pDVMPCFFModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVMPConvolCoeffFunctionModule(
                    PARTONS::DVMPCFFGK06::classId);

    // Create parameters to configure later DVMPCFFModel with PerturbativeQCD = LO
    ElemUtils::Parameters parameters;

    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::PerturbativeQCDOrderType::PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
                    PARTONS::PerturbativeQCDOrderType::LO));

    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::DVMPCFFGK06::PARAMETER_NAME_DVMPCFFGK06_MC_NWARMUP,
                    10000));
    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::DVMPCFFGK06::PARAMETER_NAME_DVMPCFFGK06_MC_NCALLS,
                    100000));
    parameters.add(
            ElemUtils::Parameter(
                    PARTONS::DVMPCFFGK06::PARAMETER_NAME_DVMPCFFGK06_MC_CHI2LIMIT,
                    0.8));

    // Configure DVMPCFFModule with previous parameters.
    pDVMPCFFModule->configure(parameters);

    // Create XiConverterModule
    PARTONS::DVMPXiConverterModule* pXiConverterModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVMPXiConverterModule(
                    PARTONS::DVMPXiConverterXBToXi::classId);

    // Create ScalesModule
    PARTONS::DVMPScalesModule* pScalesModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVMPScalesModule(
                    PARTONS::DVMPScalesQ2Multiplier::classId);

    // Set its lambda parameter, so MuF2 = MuR2 = lambda * Q2
    pScalesModule->configure(
            ElemUtils::Parameter(
                    PARTONS::DVMPScalesQ2Multiplier::PARAMETER_NAME_LAMBDA,
                    1.));

    // Create ProcessModule
    PARTONS::DVMPProcessModule* pProcessModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVMPProcessModule(
                    PARTONS::DVMPProcessGK06::classId);

    // Create Observable
    PARTONS::DVMPObservable* pObservable =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newDVMPObservable(
                    PARTONS::DVMPCrossSectionUUUMinus::classId);

    // Link modules (set physics assumptions of your computation)
    pObservable->setProcessModule(pProcessModule);
    pProcessModule->setScaleModule(pScalesModule);
    pProcessModule->setXiConverterModule(pXiConverterModule);
    pProcessModule->setConvolCoeffFunctionModule(pDVMPCFFModule);
    pDVMPCFFModule->setGPDModule(pGPDModule);



    // Load list of kinematics from file
    PARTONS::DVMPObservableKinematic observableKinematic =
            PARTONS::DVMPObservableKinematic(0.2, -0.1, 2., 6., 0.,
                    PARTONS::MesonType::PI0);

    // Create kinematic
    PARTONS::DVMPObservableResult observableResult =
            pObservableService->computeSingleKinematic(observableKinematic,
                    pObservable);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            observableResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModule, 0);
    pGPDModule = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pDVMPCFFModule, 0);
    pDVMPCFFModule = 0;

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

void changeIntegrationRoutine() {

    // Retrieve GPD service
    PARTONS::GPDService* pGPDService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getGPDService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16Numerical::classId);

    // Change integration routine
    pGPDModel->configure(
            ElemUtils::Parameter(
                    PARTONS::MathIntegratorModule::PARAM_NAME_INTEGRATOR_TYPE,
                    NumA::IntegratorType1D::GK21_ADAPTIVE));

    // Create a GPDKinematic(x, xi, t, MuF2, MuR2) to compute
    PARTONS::GPDKinematic gpdKinematic(0.1, 0.2, -0.1, 2., 2.);

    // Run computation
    PARTONS::GPDResult gpdResult = pGPDService->computeSingleKinematic(
            gpdKinematic, pGPDModel);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            gpdResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModel, 0);
    pGPDModel = 0;
}

void makeUseOfGPDEvolution() {

    // Retrieve GPD service
    PARTONS::GPDService* pGPDService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getGPDService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Create GPD evolution module with the BaseModuleFactory
    PARTONS::GPDEvolutionModule* pGPDEvolutionModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDEvolutionModule(
                    PARTONS::GPDEvolutionVinnikov::classId);

    // Create alphaS module with the BaseModuleFactory
    PARTONS::RunningAlphaStrongModule* pRunningAlphaStrongModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newRunningAlphaStrongModule(
                    PARTONS::RunningAlphaStrongVinnikov::classId);

    // Create active flavors thresholds module with the BaseModuleFactory
    PARTONS::ActiveFlavorsThresholdsModule* pActiveFlavorsThresholdsModule =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newActiveFlavorsThresholdsModule(
                    PARTONS::ActiveFlavorsThresholdsConstant::classId);

    // ActiveFlavorsThresholdsConstant module allows you to set the desired nf value that will be constant during performing the evolution.
    // Default value is nf = 3. You can change it in the following way, but you must be sure that both used GPD model and evolution routine can handle it.
    static_cast<PARTONS::ActiveFlavorsThresholdsConstant*>(pActiveFlavorsThresholdsModule)->setNFlavors(
            3);

    // Create parameters to configure later GPDEvolutionModule
    ElemUtils::Parameters parameters;

    // Number of steps in the factorization scale (i.e. set the number of steps in the integration over factorization scale)
    // One step is a typical value for Vinnikov code
    parameters.add(NumA::QuadratureIntegrator1D::PARAM_NAME_N, 2);

    // PerturbativeQCD = LO
    parameters.add(
            PARTONS::PerturbativeQCDOrderType::PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO);

    // Configure GPDEvolutionModule with previous parameters.
    pGPDEvolutionModel->configure(parameters);

    // Link modules (set physics assumptions of your computation)
    pGPDEvolutionModel->setRunningAlphaStrongModule(pRunningAlphaStrongModule);
    pGPDEvolutionModel->setActiveFlavorsModule(pActiveFlavorsThresholdsModule);
    pGPDModel->setEvolQcdModule(pGPDEvolutionModel);

    // Create a GPDKinematic(x, xi, t, MuF2, MuR2) to compute
    PARTONS::GPDKinematic gpdKinematic(0.1, 0.2, -0.1, 40., 40.);

    // Run computation
    PARTONS::GPDResult gpdResult = pGPDService->computeSingleKinematic(
            gpdKinematic, pGPDModel);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            gpdResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pActiveFlavorsThresholdsModule, 0);
    pGPDModel = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pRunningAlphaStrongModule, 0);
    pGPDModel = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDEvolutionModel, 0);
    pGPDModel = 0;

    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModel, 0);
    pGPDModel = 0;
}

void selectSpecificGPDTypes() {

    // Retrieve GPD service
    PARTONS::GPDService* pGPDService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getGPDService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Create a list of GPD types you want to compute for
    PARTONS::List<PARTONS::GPDType> gpdTypelist;

    gpdTypelist.add(PARTONS::GPDType::E);
    gpdTypelist.add(PARTONS::GPDType::Et);

    // Create a GPDKinematic(x, xi, t, MuF2, MuR2) to compute
    PARTONS::GPDKinematic gpdKinematic(0.1, 0.2, -0.1, 2., 2.);

    // Run computation
    PARTONS::GPDResult gpdResult = pGPDService->computeSingleKinematic(
            gpdKinematic, pGPDModel, gpdTypelist);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            gpdResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModel, 0);
    pGPDModel = 0;
}

void demonstrateUnits() {

    // Retrieve GPD service
    PARTONS::GPDService* pGPDService =
            PARTONS::Partons::getInstance()->getServiceObjectRegistry()->getGPDService();

    // Create GPD module with the BaseModuleFactory
    PARTONS::GPDModule* pGPDModel =
            PARTONS::Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    PARTONS::GPDGK16::classId);

    // Kinematics
    PARTONS::PhysicalType<double> x(0.1, PARTONS::PhysicalUnit::NONE);
    PARTONS::PhysicalType<double> xi(0.2, PARTONS::PhysicalUnit::NONE);
    PARTONS::PhysicalType<double> t(-0.1, PARTONS::PhysicalUnit::GEV2);
    PARTONS::PhysicalType<double> muF2(2., PARTONS::PhysicalUnit::GEV2);
    PARTONS::PhysicalType<double> muR2(2., PARTONS::PhysicalUnit::GEV2);

    PARTONS::PhysicalType<double> tInMeV2_a = t.makeSameUnitAs(
            PARTONS::PhysicalUnit::MEV2);

    PARTONS::PhysicalType<double> tInMeV2_b = t;
    tInMeV2_b.makeSameUnitAs(tInMeV2_a);

    // Create a GPDKinematic(x, xi, t, MuF2, MuR2) to compute
    PARTONS::GPDKinematic gpdKinematic(x, xi, tInMeV2_a, muF2, muF2);

    // Run computation
    PARTONS::GPDResult gpdResult = pGPDService->computeSingleKinematic(
            gpdKinematic, pGPDModel);

    // Print results
    PARTONS::Partons::getInstance()->getLoggerManager()->info("main", __func__,
            gpdResult.toString());

    // Remove pointer references
    // Module pointers are managed by PARTONS
    PARTONS::Partons::getInstance()->getModuleObjectFactory()->updateModulePointerReference(
            pGPDModel, 0);
    pGPDModel = 0;
}
