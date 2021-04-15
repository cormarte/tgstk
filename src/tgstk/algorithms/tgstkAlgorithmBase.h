/*==========================================================================

  Program:   Tumour Growth Simulation Toolkit
  Module:    tgstkAlgorithmBase.h

  Copyright (c) Corentin Martens
  All rights reserved.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
     NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
     ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR
     OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING
     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
     OTHER DEALINGS IN THE SOFTWARE.

==========================================================================*/

/**
 *
 * @class tgstkAlgorithmBase
 *
 * @brief Base class for TGSTK algorithms.
 *
 * tgstkAlgorithmBase is a base class for TGSTK algorithms.
 *
 */

#ifndef TGSTKALGORITHMBASE_H
#define TGSTKALGORITHMBASE_H

#define assertNotNullPtr(var) _assertNotNullPtr(var, #var)
#define assertValueInRange(var, min, max) _assertValueInRange(var, min, max, #var)
#define assertValueIsEqual(var1, var2) _assertValueIsEqual(var1, var2, #var1, #var2)

#include <tgstk/tgstkGlobal.h>
#include <tgstk/core/tgstkObjectBase.h>

class TGSTK_EXPORT tgstkAlgorithmBase : public tgstkObjectBase {

    public:

        virtual ~tgstkAlgorithmBase();

        virtual bool check();
        virtual void execute()=0;

        bool update();

    protected:

        tgstkAlgorithmBase();

        template<typename Type> bool _assertNotNullPtr(Type var, std::string name);
        template<typename Type> bool _assertValueInRange(Type var, Type min, Type max, std::string name);
        template<typename Type1, typename Type2> bool _assertValueIsEqual(Type1 var1, Type2 var2, std::string name1, std::string name2);
};

template<typename Type>
bool tgstkAlgorithmBase::_assertNotNullPtr(Type var, std::string name) {

    bool assert = var != nullptr;

    if (!assert) {

        cout << this->objectName << ": Error: '" << name << "' is nullptr." << endl;
    }

    return assert;
}

template<typename Type>
bool tgstkAlgorithmBase::_assertValueInRange(Type var, Type min, Type max, std::string name) {

    bool assert = (var >= min && var <= max);

    if (!assert) {

        cout << this->objectName << ": Error: '" << name << "' should be in range [" << min << "; " << max <<"]." << endl;
    }

    return assert;
}

template<typename Type1, typename Type2>
bool tgstkAlgorithmBase::_assertValueIsEqual(Type1 var1, Type2 var2, std::string name1, std::string name2) {

    bool assert = (var1 == var2);

    if (!assert) {

        cout << this->objectName << ": Error: '" << name1 << "' not equal to '" << name2 << "'." << endl;
    }

    return assert;
}

#endif // TGSTKALGORITHMBASE_H
