#ifndef __SNP_ERROR_CODE_H__
#define __SNP_ERROR_CODE_H__

#include <snp-cuda\snpMacros.h>

NS_SNP_BEGIN

class snpErrorCode
{
public:
	typedef enum
	{
		SUCCEEDED = 0,

		DEVICE_ALREADY_CONFIGURED,
		DEVICE_NOT_CONFIGURED,
		GPU_INIT_ERROR

	} enum_type;

private:
	const enum_type m_value;

public:
	snpErrorCode(enum_type value)
		: m_value(value)
	{
		// don't forget to update assert
		snpAssert(m_value >= SUCCEEDED && m_value <= GPU_INIT_ERROR);
	}

	operator enum_type() const
	{
		return m_value;
	}
};

NS_SNP_END

#endif //__SNP_ERROR_CODE_H__
