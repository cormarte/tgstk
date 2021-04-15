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
