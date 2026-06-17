//
// Created by Mariana Khachatryan on 6/15/26.
//

#ifndef DVCS_OBSERVABLE_SERVICE_TORCH_H
#define DVCS_OBSERVABLE_SERVICE_TORCH_H

#include <partons/services/DVCSObservableService.h>

#include <string>

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h"
#include "NNFit/Theory/Modules/Services/ObservableServiceTorch.h"

/**
 * @class DVCSObservableServiceTorch
 *
 * @brief Torch-aware sibling of PARTONS::DVCSObservableService.
 *
 * Registered in the PARTONS BaseObjectRegistry exactly like the scalar service
 * (protected-ctor self-registration), so it is retrievable through the standard
 * ServiceObjectRegistry by name:
 *
 *   auto* s = static_cast<DVCSObservableServiceTorch*>(
 *       Partons::getInstance()->getServiceObjectRegistry()
 *               ->get("DVCSObservableServiceTorch"));
 *
 * It inherits the full scalar machinery untouched (computeSingleKinematic and
 * the XML-task helpers keep returning DVCSObservableResult) and gains the
 * generic tensor driver computeSingleKinematicTorch() from the channel-agnostic
 * ObservableServiceTorch<DVCSObservableKinematic> mixin — the differentiable
 * counterpart of ObservableService::computeSingleKinematic(). The driver takes a
 * base DVCSObservableTorch*, so it drives any DVCS tensor observable
 * polymorphically.
 */
class DVCSObservableServiceTorch: public PARTONS::DVCSObservableService,
        public ObservableServiceTorch<PARTONS::DVCSObservableKinematic> {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    virtual ~DVCSObservableServiceTorch();

protected:

    /**
     * Constructor (protected, called by the registry self-registration).
     * @param className Name of last child class.
     */
    DVCSObservableServiceTorch(const std::string& className);
};

#endif /* DVCS_OBSERVABLE_SERVICE_TORCH_H */