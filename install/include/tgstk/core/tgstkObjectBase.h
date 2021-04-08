#ifndef TGSTKOBJECTBASE_H
#define TGSTKOBJECTBASE_H

# include <tgstk/tgstkGlobal.h>

#include <string>

class TGSTK_EXPORT tgstkObjectBase {

    public:

        virtual ~tgstkObjectBase();

        std::string getObjectName();

    protected:

        tgstkObjectBase();

        std::string objectName;
};

#endif // TGSTKOBJECTBASE_H
