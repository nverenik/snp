#ifndef __SNP_OPERATION_H__
#define __SNP_OPERATION_H__

#include <snp/Macros.h>

NS_SNP_BEGIN

enum tOperation
{
    tOperation_Assign   = 0x00,
    tOperation_Not      = 0x01,
    tOperation_And      = 0x02,
    tOperation_Or       = 0x03
};

NS_SNP_END

#endif //__SNP_OPERATION_H__
