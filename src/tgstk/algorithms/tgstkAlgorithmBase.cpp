#include <tgstk/algorithms/tgstkAlgorithmBase.h>

#include <iostream>

tgstkAlgorithmBase::tgstkAlgorithmBase() {

}

tgstkAlgorithmBase::~tgstkAlgorithmBase() {

}

bool tgstkAlgorithmBase::check() {

    std::cout << this->objectName << ": Warning: check() method not reimplemented." << std::endl;

    return true;
}

bool tgstkAlgorithmBase::update() {

    if (this->check()) {

        this->execute();
        return true;
    }

    return false;
}
