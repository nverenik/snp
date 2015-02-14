#include "RenameMe.h"
#include "DataTypes.h"

int SetSocketOptions(int iSocketFD, int iLevel, int iOptName, const void *pOptVal, int optlen)
{
#ifdef WIN32
    return setsockopt(iSocketFD, iLevel, iOptName, (const char *)pOptVal, optlen);
#else
    return setsockopt(iSocketFD, iLevel, iOptName, pOptVal, optlen);
#endif // WIN32
}

int CloseSocket(int iSocketFD)
{
#ifdef WIN32
    return closesocket(iSocketFD);
#else
    return close(iSocketFD);
#endif // WIN32
}

int ReadFromSocket(int iSocketFD, void *pBuffer, size_t iCount)
{
#ifdef WIN32
    return recv(iSocketFD, (char *)pBuffer, iCount, 0);
#else
    return read(iSocketFD, pBuffer, iCount);
#endif // WIN32
}

int WriteToSocket(int iSocketFD, const void *pBuffer, size_t iCount)
{
#ifdef WIN32
    return send(iSocketFD, (const char *)pBuffer, iCount, 0);
#else
    return write(iSocketFD, pBuffer, iCount);
#endif // WIN32
}

void msleep(int milliseconds)
{
#ifdef WIN32
    Sleep(milliseconds);
#else // WIN32
    usleep(static_cast<useconds_t>(milliseconds) * 1000);
#endif // WIN32
}