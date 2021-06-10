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
 * @class tgstkMeshProcessorBase
 *
 * @brief Base class for TGSTK mesh processors.
 *
 * tgstkMeshProcessorBase is a base class for TGSTK mesh processing
 * algorithms.
 *
 */

#ifndef TGSTKMESHPROCESSORBASE_H
#define TGSTKMESHPROCESSORBASE_H

#include <tgstk/tgstkGlobal.h>
#include <tgstk/algorithms/tgstkAlgorithmBase.h>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

class TGSTK_EXPORT tgstkMeshProcessorBase : public virtual tgstkAlgorithmBase {

    public:

        virtual ~tgstkMeshProcessorBase();

    protected:

        tgstkMeshProcessorBase();
};

#endif // TGSTKMESHPROCESSORBASE_H
