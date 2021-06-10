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

%{
#include <vtkObjectBase.h>
#include <vtkPythonUtil.h>
#include <vtkSmartPointer.h>
%}

%include "exception.i"

%define VTK_SWIG_INTEROP(VTK_TYPE)

%{
#include <VTK_TYPE##.h>
%}

%typemap(out) VTK_TYPE* {

	PyImport_ImportModule("vtk");
	$result = vtkPythonUtil::GetObjectFromPointer(static_cast<vtkObjectBase*>($1));
}

%typemap(out) vtkSmartPointer<VTK_TYPE> {
	
	PyImport_ImportModule("vtk");
	$result = vtkPythonUtil::GetObjectFromPointer(static_cast<vtkObjectBase*>($1));
}

%typemap(in) VTK_TYPE* {

    $1 = static_cast<VTK_TYPE*>(vtkPythonUtil::GetPointerFromObject($input,#VTK_TYPE));

    if (!$1) {

        SWIG_exception(SWIG_TypeError, "A " #VTK_TYPE " object is required.");
    }
}

%typemap(in) vtkSmartPointer<VTK_TYPE> {

    $1 = vtkSmartPointer<VTK_TYPE>(static_cast<VTK_TYPE*>(vtkPythonUtil::GetPointerFromObject($input,#VTK_TYPE)));

    if (!$1) {

        SWIG_exception(SWIG_TypeError, "A " #VTK_TYPE " object is required.");
    }
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) VTK_TYPE* {

  $1 = vtkPythonUtil::GetPointerFromObject($input,#VTK_TYPE) ? 1 : 0;
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) vtkSmartPointer<VTK_TYPE> {

  $1 = vtkPythonUtil::GetPointerFromObject($input,#VTK_TYPE) ? 1 : 0;
}

%enddef
