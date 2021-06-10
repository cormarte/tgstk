/*==========================================================================

  This file is part of the Tumor Growth Simulation ToolKit (TGSTK)
  (<https://github.com/cormarte/TGSTK>, <https://cormarte.github.io/TGSTK>).

  Copyright (C) 2021  Corentin Martens

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <https://www.gnu.org/licenses/>.

  Contact: corentin.martens@ulb.be

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
