#ifndef __ERROR_CODE_H__
#define __ERROR_CODE_H__

#include <snp/Macros.h>

NS_SNP_BEGIN

class CErrorCode
{
public:
    typedef enum
    {
        UNDEFINED = 0,
        SUCCEEDED,

        DEVICE_ALREADY_CONFIGURED,
        DEVICE_NOT_CONFIGURED,
        GPU_INIT_ERROR

    } enum_type;

private:
    enum_type m_value;

public:
    CErrorCode(enum_type value = UNDEFINED)
        : m_value(value)
    {
        // don't forget to update assert
        snpAssert(m_value >= SUCCEEDED && m_value <= GPU_INIT_ERROR);
    }

    operator enum_type() const
    {
        return m_value;
    }

    CErrorCode & operator =(const CErrorCode &other)
    {
        (*this).m_value = other.m_value;
        return (*this);
    }
};

NS_SNP_END

#endif //__ERROR_CODE_H__
