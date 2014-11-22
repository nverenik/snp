#ifndef __SNP_OPERATION_H__
#define __SNP_OPERATION_H__

#include <snp\snpMacros.h>

NS_SNP_BEGIN

enum snpOperation
{
	snpAssign	= 0x00,
	snpNot		= 0x01,
	snpAnd		= 0x02,
	snpOr		= 0x03
};

NS_SNP_END

#endif //__SNP_OPERATION_H__
