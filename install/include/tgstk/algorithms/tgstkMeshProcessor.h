#ifndef TGSTKMESHPROCESSOR_H
#define TGSTKMESHPROCESSOR_H

#include <tgstk/tgstkGlobal.h>
#include <tgstk/algorithms/tgstkAlgorithmBase.h>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

class TGSTK_EXPORT tgstkMeshProcessor : public virtual tgstkAlgorithmBase {

    public:

        virtual ~tgstkMeshProcessor();

    protected:

        tgstkMeshProcessor();
};

#endif // TGSTKMESHPROCESSOR_H
