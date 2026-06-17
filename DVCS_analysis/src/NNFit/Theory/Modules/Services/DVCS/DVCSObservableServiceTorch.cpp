//
// Created by Mariana Khachatryan on 6/15/26.
//

#include "NNFit/Theory/Modules/Services/DVCS/DVCSObservableServiceTorch.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>

const unsigned int DVCSObservableServiceTorch::classId =
        PARTONS::Partons::getInstance()->getBaseObjectRegistry()->registerBaseObject(
                new DVCSObservableServiceTorch("DVCSObservableServiceTorch"));

DVCSObservableServiceTorch::DVCSObservableServiceTorch(
        const std::string& className) :
        PARTONS::DVCSObservableService(className) {
}

DVCSObservableServiceTorch::~DVCSObservableServiceTorch() {
}