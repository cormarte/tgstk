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
