#ifndef _SOCKET_H_
#define _SOCKET_H_

#include "DataTypes.h"

#define SOCKET_IN_BUFFER_MAX_SIZE	65530
#define SOCKET_OUT_BUFFER_MAX_SIZE	65530
#define SENDING_MAX_SIZE			65530

struct tSocket
{
	int m_iSocketId;
	int m_iSocketHandle;
	time_t m_iConnectTime;
	bool m_bClosed;
	bool m_bExtract;

	BYTE m_aReadBuffer[SENDING_MAX_SIZE];

	vector<BYTE> m_abtInBuffer;
	vector<BYTE> m_abtOutBuffer;

	tSocket();
	int Read();
	int Send();

};

#endif // _SOCKET_H_
