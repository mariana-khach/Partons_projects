//
// Created by Mariana Khachatryan on 3/5/26.
//

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/logger/LoggerManager.h>
#include <ElementaryUtils/parameters/Parameter.h>
#include <partons/beans/KinematicUtils.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/observable/ObservableResult.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSCFFNN.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSCFFNNReplicas.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUMinusPhiIntegrated.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinusSin1Phi.h>
#include <partons/modules/process/DVCS/DVCSProcessBMJ12.h>
#include <partons/modules/scales/DVCS/DVCSScalesQ2Multiplier.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterXBToXi.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/services/ObservableService.h>
#include <partons/ServiceObjectRegistry.h>
#include <partons/utils/type/PhysicalType.h>
#include <stddef.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

double getMean(const std::vector<double>& v) {

    if (v.size() == 0) {
        std::cout << __func__ << ": vector size is 0" << std::endl;
        exit(0);
    }

    double mean = 0.;

    for (int i = 0; i < v.size(); i++) {
        mean += v.at(i);
    }

    return mean / v.size();
}

double getSigma(const std::vector<double>& v) {

    if (v.size() == 0) {
        std::cout << __func__ << ": vector size is 0" << std::endl;
        exit(0);
    }

    double mean = getMean(v);

    double sigma = 0.;

    for (int i = 0; i < v.size(); i++) {
        sigma += pow(mean - v.at(i), 2);
    }

    return sqrt(sigma / double(v.size()));
}

size_t removeOutliers(std::vector<double>& v) {

    if (v.size() == 0) {
        std::cout << __func__ << ": vector size is 0" << std::endl;
        exit(0);
    }

    double meanData = getMean(v);
    double sigmaData = getSigma(v);

    if (sigmaData == 0.) {
        std::cout << __func__ << ": warning: sigma is 0" << std::endl;
        return 0;
    }

    std::vector<double> result;

    std::vector<double>::iterator it;
    size_t nRemoved = 0;

    for (it = v.begin(); it != v.end(); it++) {

        if (fabs((*it) - meanData) / sigmaData > 3.) {
            nRemoved++;
        } else {
            result.push_back(*it);
        }
    }

    v = result;

    if (nRemoved != 0)
        nRemoved += removeOutliers(v);

    return nRemoved;
}

void analysisANN_SingleKin() {

    //namespace
    using namespace PARTONS;

    //modules
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNN::classId);

    DVCSXiConverterModule* pDVCSXiConverter =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSXiConverterModule(
                    DVCSXiConverterXBToXi::classId);

    DVCSScalesModule* pDVCSScales =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSScalesModule(
                    DVCSScalesQ2Multiplier::classId);

    DVCSProcessModule* pDVCSProcess =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSProcessModule(
                    DVCSProcessBMJ12::classId);

    DVCSObservable* pDVCSObs =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    DVCSCrossSectionUUMinusPhiIntegrated::classId);

    //pQCD order
    pDVCSCFF->setQCDOrderType(PerturbativeQCDOrderType::LO);

    //link modules
    pDVCSProcess->setXiConverterModule(pDVCSXiConverter);
    pDVCSProcess->setScaleModule(pDVCSScales);
    pDVCSProcess->setConvolCoeffFunctionModule(pDVCSCFF);
    pDVCSObs->setProcessModule(pDVCSProcess);

    //service
    DVCSObservableService* pObservableService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();

    //kineamtics
    DVCSObservableKinematic dvcsKinematics(0.2, -0.2, 2., 12., 0.);

    //evaluate
    std::vector<double> results;

    for (size_t i = 0; i < c_nDVCSCFFNNReplicas; i++) {

        pDVCSProcess->resetPreviousKinematic();

        static_cast<DVCSCFFNN*>(pDVCSCFF)->configure(
                ElemUtils::Parameter(DVCSCFFNN::PARAMETER_NAME_REPLICA, i));

        results.push_back(
                pObservableService->computeSingleKinematic(dvcsKinematics,
                        pDVCSObs).getValue().getValue());
    }

    //remove outliers
    removeOutliers(results);

    std::cout << results.size() << std::endl;

    std::cout << "Observable name: " << pDVCSObs->getClassName() << std::endl;
    std::cout << "Kinematics: " << dvcsKinematics.toString() << std::endl;
    std::cout << "0.68 confidence level: "
            << getMean(results) - getSigma(results) << " -- "
            << getMean(results) + getSigma(results) << std::endl;

    return;
}












void analysisANN_ManyKin() {

    //namespace
    using namespace PARTONS;

    //modules
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNN::classId);

    DVCSXiConverterModule* pDVCSXiConverter =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSXiConverterModule(
                    DVCSXiConverterXBToXi::classId);

    DVCSScalesModule* pDVCSScales =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSScalesModule(
                    DVCSScalesQ2Multiplier::classId);

    DVCSProcessModule* pDVCSProcess =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSProcessModule(
                    DVCSProcessBMJ12::classId);

    DVCSObservable* pDVCSObs =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    DVCSAluMinusSin1Phi::classId);

    //pQCD order
    pDVCSCFF->setQCDOrderType(PerturbativeQCDOrderType::LO);

    //link modules
    pDVCSProcess->setXiConverterModule(pDVCSXiConverter);
    pDVCSProcess->setScaleModule(pDVCSScales);
    pDVCSProcess->setConvolCoeffFunctionModule(pDVCSCFF);
    pDVCSObs->setProcessModule(pDVCSProcess);

    //service
    DVCSObservableService* pObservableService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();

    // Load list of kinematics from file
    List<DVCSObservableKinematic> observableKinematicList =
            KinematicUtils().getDVCSObservableKinematicFromFile(
                    "/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_input/BSA_CLAS_15_KK_format.csv");

    // One results vector per kinematic point, accumulating over replicas
    std::vector<std::vector<double>> results(observableKinematicList.size());

    //evaluate
    for (size_t i = 0; i < c_nDVCSCFFNNReplicas; i++) {

        pDVCSProcess->resetPreviousKinematic();

        static_cast<DVCSCFFNN*>(pDVCSCFF)->configure(
                ElemUtils::Parameter(DVCSCFFNN::PARAMETER_NAME_REPLICA, i));

        List<DVCSObservableResult> observableResultList =
                pObservableService->computeManyKinematic(observableKinematicList,
                        pDVCSObs);

        for (size_t j = 0; j < observableKinematicList.size(); j++) {
            results.at(j).push_back(observableResultList[j].getValue().getValue());
        }
    }

    // Write results to CSV
    std::ofstream out("/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_output/dvcs_DVCSAluSinPhi_BSACLAS15_ANN.csv");
    std::ofstream out1("/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_output/dvcs_DVCSAluSinPhi_ANN_replicas.csv");
    out << "xB,t,Q2,E,phi,mean,sigma\n";

    for (size_t j = 0; j < observableKinematicList.size(); j++) {

        removeOutliers(results.at(j));

        const auto& kin = observableKinematicList[j];

        out << std::fixed << std::setprecision(6)
            << kin.getXB().getValue()  << ","
            << kin.getT().getValue()   << ","
            << kin.getQ2().getValue()  << ","
            << kin.getE().getValue()   << ","
            << kin.getPhi().getValue() << ","
            << getMean(results.at(j))  << ","
            << getSigma(results.at(j)) << "\n";


    if (j == 3) {
    for (const double& val : results.at(j)) {
        out1 << std::fixed << std::setprecision(6) << val << "\n";
    }
    }

    }

    out.close();
    out1.close();

    return;
}











int main(int argc, char** argv) {

    PARTONS::Partons* pPartons = 0;

    try {

        pPartons = PARTONS::Partons::getInstance();
        pPartons->init(argc, argv);

        //analysisANN_SingleKin();
        analysisANN_ManyKin();

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
