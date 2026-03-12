#ifndef EXAMPLES_H_
#define EXAMPLES_H_

// ******************************************************
// GPD EXAMPLES *****************************************
// ******************************************************

/**
 * This function demonstrates a simple task as the evaluation of collinear distributions in a single kinematic point.
 * The result of this function is printed out to the standard output.
 * These are the values for all defined in the collinear distributions, including singlet and non-singlet combinations for quarks.
 */
void computeSingleKinematicsForCollinearDistribution();

/**
 * This function demonstrates a simple task as the evaluation of GPD model in a single kinematic point.
 * The result of this function is printed out to the standard output.
 * These are the values for all defined in the model GPDs, including singlet and non-singlet combinations for quarks.
 */
void computeSingleKinematicsForGPD();

/**
 * This function demonstrates the evaluation of GPD model for kinematics defined in a text file.
 * In this file kinematic points are encoded in separate lines using the following format: "x|xi|t|MuF2|MuR2".
 * The result of this function is printed out to the standard output.
 * These are the values for all defined in the model GPDs, including singlet and non-singlet combinations for quarks.
 */
void computeManyKinematicsForGPD();

// ******************************************************
// CFF EXAMPLES *****************************************
// ******************************************************

/**
 * This function demonstrates a simple task as the evaluation of DVCS Compton Form Factors (CFF) in a single kinematic point.
 * The result of this function is printed out to the standard output.
 * These are CFF values for all GPD types defined in the selected GPD model.
 */
void computeSingleKinematicsForDVCSComptonFormFactor();

/**
 * This function demonstrates the evaluation of DVCS Compton Form Factors (CFF) for kinematics defined in a text file.
 * In this file kinematic points are encoded in separate lines using the following format: "xi|t|Q2|MuF2|MuR2".
 * The result of this function is printed out to the standard output.
 * These are CFF values for all GPD types defined in the selected GPD model.
 */
void computeManyKinematicsForDVCSComptonFormFactor();

/**
 * This function demonstrates a simple task as the evaluation of DVMP Compton Form Factors (CFF) in a single kinematic point.
 * The result of this function is printed out to the standard output.
 * These are CFF values for all GPD types defined in the selected GPD model.
 */
void computeSingleKinematicsForDVMPComptonFormFactor();

// ******************************************************
// OBSERVABLE EXAMPLES **********************************
// ******************************************************

/**
 * This function demonstrates a simple task as the evaluation of DVCS observable in a single kinematic point.
 * The result of this function is printed out to the standard output.
 */
void computeSingleKinematicsForDVCSObservable();

/**
 * This function demonstrates the evaluation of DVCS observable for kinematics defined in a text file.
 * In this file kinematic points are encoded in separate lines using the following format: "xB|t|Q2|E|phi".
 * The result of this function is printed out to the standard output.
 */
void computeManyKinematicsForDVCSObservable();

/**
 * This function demonstrates a simple task as the evaluation of DVMP observable in a single kinematic point.
 * The result of this function is printed out to the standard output.
 */
void computeSingleKinematicsForDVMPObservable();

// ******************************************************
// OTHER ************************************************
// ******************************************************

/**
 * This function demonstrates how to change the integration routine in one of modules.
 * To make it possible, the module must inherit from MathIntegratorModule class (our doxygen documentation will tell you that).
 *
 * Be careful when you perform this operation.
 * Some integration routines may speed up computations, but at the same time they can be not accurate enough to be applied in some kinematic ranges (e.g. in low xB).
 * The infinities are also treated differently by various integration routines (or they are not treated at all).
 *
 * This function is the demonstration for GPD module.
 * Note however that the way of changing the integration routine that is presented here is applicable to any type of PARTONS module.
 */
void changeIntegrationRoutine();

/**
 * This function demonstrates a simple task as the evaluation of GPD model in a single kinematic point making use of GPD evolution.
 * The result of this function is printed out to the standard output.
 * These are the values for all defined in the model GPDs, including singlet and non-singlet combinations for quarks.
 * Note that you can use this example to include the GPD evolution in the evaluation of CFFs and observables.
 */
void makeUseOfGPDEvolution();

/**
 * This function demonstrates a simple task as the evaluation of GPD model in a single kinematic point for specific GPD types only.
 * The result of this function is printed out to the standard output.
 * These are the values for all defined in the model GPDs, including singlet and non-singlet combinations for quarks.
 * Note that you can use this example to select specific GPD types in the evaluation of CFFs and observables.
 */
void selectSpecificGPDTypes();

/**
 * This function demonstrates the usage of units.
 */
void demonstrateUnits();


#endif /* INCLUDE_EXAMPLES_H_ */
