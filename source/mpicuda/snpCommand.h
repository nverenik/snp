#ifndef __SNP_COMMAND_H__
#define __SNP_COMMAND_H__

#include <snp/snpMacros.h>

NS_SNP_BEGIN

//enum class snpCommand : int32
//{
//	SYSTEM_INFO,
//	STARTUP,
//	SHUTDOWN,
//	EXEC,
//	READ
//};

enum snpCommand
{
	Undefined	= -1,
	SYSTEM_INFO	= 0,	// for now without this command
	STARTUP		= 1,
	SHUTDOWN	= 2,
	EXEC		= 3,
	READ		= 4
};

NS_SNP_END

#endif //__SNP_COMMAND_H__
