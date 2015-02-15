#ifndef __SNP_MACROS_H__
#define __SNP_MACROS_H__

#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
    #define NS_SNP_BEGIN    namespace snp {
    #define NS_SNP_END      }
    #define USING_NS_SNP    using namespace snp
#else
    #define NS_SNP_BEGIN
    #define NS_SNP_END
    #define USING_NS_SNP
#endif

#define snpNotUsed(__x__)   ((void)(__x__))
#define snpAssert(__A__)    assert(__A__)

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef signed char int8;
typedef signed short int16;
typedef signed int int32;
typedef signed long long int64;

typedef float float32;
typedef double float64;

#ifndef MIN
#define MIN(__x__, __y__) (((__x__) < (__y__)) ? (__x__) : (__y__))
#endif // MIN

#ifndef MAX
#define MAX(__x__, __y__) (((__x__) > (__y__)) ? (__x__) : (__y__))
#endif // MAX

#define snpBit(__x__) (1 << (__x__))

#define snpCompareBits(__data__, __mask__, __bitfield__) ((((__mask__) & (__data__)) ^ ((__mask__) & (__bitfield__))) == 0)
#define snpUpdateBitsASSIGN(__data__, __mask__, __bitfield__) { (__data__) = ((~(__mask__) & (__data__)) | ((__mask__) & (__bitfield__))); }
#define snpUpdateBitsNOT(__data__, __mask__, __bitfield__) { (__data__) = (((__mask__) | (__data__)) & (~(__mask__) | ~(__data__))); }
#define snpUpdateBitsAND(__data__, __mask__, __bitfield__) { (__data__) = ((__data__) & (~(__mask__) | (__bitfield__))); }
#define snpUpdateBitsOR(__data__, __mask__, __bitfield__) { (__data__) = ((__data__) | ((__mask__) & (__bitfield__))); }

#define snpUpdateBits(__operation__, __data__, __mask__, __bitfield__, __cellSize__) \
    for (uint32 index = 0; index < (__cellSize__); index++) { \
        __operation__((__data__), (__mask__), (__bitfield__)) \
    }

#define snpBitfieldSet(__bitfield__, __value__) { \
    uint16 size = sizeof(__bitfield__) / sizeof((__bitfield__)[0]); \
    for (uint16 index = 0; index < size; index++) { \
        (__bitfield__)[index] = (__value__); \
    } \
}

#endif //__SNP_MACROS_H__
