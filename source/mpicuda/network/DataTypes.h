#ifndef _DATATYPES_H_
#define _DATATYPES_H_

#ifndef WIN32
#include <linux/stddef.h>
#include <linux/types.h>
#include <linux/elf.h>
#include <sys/time.h>
#include <errno.h>
#include <string.h>
#endif // !WIN32

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <iostream>
#include <assert.h>
#include <string>
#include <vector>
#include <list>

#ifdef WIN32
//#define FD_SETSIZE  1024
#include <winsock2.h>
#include <WS2tcpip.h>
#include <windows.h>
#else // WIN32
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#endif // WIN32

#include <snp/Macros.h>

#define finline inline

#ifndef WIN32
typedef uint8 BYTE;
typedef uint16 WORD;
typedef uint32 DWORD;
//typedef __u8  UCHAR, UINT8, BYTE;
//typedef __u16 USHORT, UINT16, WORD;
//typedef __u32 UINT, UINT32, DWORD;

#endif // !WIN32

template <class TYPE> finline TYPE min2(TYPE x,TYPE y) { return ((x)<(y)?(x):(y)); }
template <class TYPE> finline TYPE max2(TYPE x,TYPE y) { return ((x)>(y)?(x):(y)); }

#define LOG_LEVEL 4

finline void LogMessage(int iLevel, const char* sFormat, ...)
{
    if(iLevel > LOG_LEVEL)
        return;

    va_list args;
    va_start(args, sFormat);
    // vprintf(sFormat, args);

    char sMessage[1024];
    vsnprintf(sMessage, 1024, sFormat, args);
    std::cout << sMessage << std::endl;

    va_end(args);
}

#define LOG_MESSAGE LogMessage

#ifndef WIN32
#define HIBYTE(w)                ((BYTE)(((WORD)(w) >> 8) & 0xFF))
#define LOBYTE(w)                ((BYTE)(w))
#define HIWORD(dw)                ((WORD)((((DWORD)(dw)) >> 16) & 0xFFFF))
#define LOWORD(dw)                ((WORD)(DWORD)(dw))
#define MAKEWORD(low, hi)        ((WORD)(((BYTE)(low)) | (((WORD)((BYTE)(hi))) << 8)))
#endif // !WIN32

#define MAKEDWORD(low, hi)        (DWORD)((low) | ((hi) << 16))

#endif // _DATATYPES_H_
