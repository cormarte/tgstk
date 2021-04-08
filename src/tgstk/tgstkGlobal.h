#ifndef TGSTKGLOBAL_H
#define TGSTKGLOBAL_H

#if defined(TGSTK_LIBRARY)
#  define TGSTK_EXPORT __declspec(dllexport)
#else
#  define TGSTK_EXPORT __declspec(dllimport)
#endif

#endif // TGSTKGLOBAL_H
