#ifndef __RENAME_ME_H__
#define __RENAME_ME_H__

#include "DataTypes.h"

int SetSocketOptions(int iSocketFD, int iLevel, int iOptName, const void *pOptVal, int optlen);
int CloseSocket(int iSocketFD);

int ReadFromSocket(int iSocketFD, void *pBuffer, size_t iCount);
int WriteToSocket(int iSocketFD, const void *pBuffer, size_t iCount);

void msleep(int milliseconds);

#endif // __RENAME_ME_H__
