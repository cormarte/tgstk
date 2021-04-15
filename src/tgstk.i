%module tgstk

%{
#include <vtkPythonUtil.h>
#include <tgstk/algorithms/tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter.h>
%}

%include "windows.i"

%include <vtk.i>
VTK_SWIG_INTEROP(vtkImageData)

%include <tgstk/tgstkGlobal.h>
%include <tgstk/algorithms/tgstkAlgorithmBase.h>
%include <tgstk/algorithms/tgstkImageProcessorBase.h>
%include <tgstk/algorithms/tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter.h>
