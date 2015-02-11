#include "ProtocolHandler.h"

CProtocolHandler::CProtocolHandler()
{
}

CProtocolHandler::~CProtocolHandler()
{
	assert(m_oSocket.m_bClosed == true);

	close(m_oSocket.m_iSocketHandle);
}

void CProtocolHandler::Tick()
{
	if(m_oSocket.m_bClosed || m_oSocket.m_iSocketHandle == -1)
		return;

	// Low level
	Select();
	ReadWrite();

	Extract();
	Pack();

	// Higher Level
	Execute();
}

void CProtocolHandler::InitSocket(int iSocketFD)
{
	m_oSocket.m_iSocketHandle = iSocketFD;
}

void CProtocolHandler::Select()
{
	int iMaxSocketHandle = m_oSocket.m_iSocketHandle;

	assert(iMaxSocketHandle < FD_SETSIZE);

	FD_ZERO(&m_oReadSet);
	FD_ZERO(&m_oWriteSet);
	FD_ZERO(&m_oErrorSet);

	if(m_oSocket.m_abtInBuffer.size() < SOCKET_IN_BUFFER_MAX_SIZE && m_aInPacketBuffer.size() < IN_PACKET_BUFFER_MAX_SIZE)
	{
		FD_SET(m_oSocket.m_iSocketHandle, &m_oReadSet);
	}

	if( !m_oSocket.m_abtOutBuffer.empty() )
	{
		FD_SET(m_oSocket.m_iSocketHandle, &m_oWriteSet);
	}

	FD_SET(m_oSocket.m_iSocketHandle, &m_oErrorSet);

	struct timeval oTimeval;
	oTimeval.tv_sec = 0;
	oTimeval.tv_usec = 1000;
	if(select(iMaxSocketHandle + 1, &m_oReadSet, &m_oWriteSet, &m_oErrorSet, &oTimeval) == -1)
	{
		LOG_MESSAGE( 1, "select error: %s", strerror(errno) );
	}
}

void CProtocolHandler::ReadWrite()
{
	// Read inbound data
	if( FD_ISSET(m_oSocket.m_iSocketHandle, &m_oReadSet) )
	{
		int iReadResult = m_oSocket.Read();
		if(iReadResult > 0)
		{
			LOG_MESSAGE(3, "Has read [%d] bytes from socket", iReadResult);
		}
		else if(iReadResult <= 0) // Close socket on error | gentle shutdown
		{
			m_oSocket.m_bClosed = true;
		}
	}

	// Send outgoing data
	if( FD_ISSET(m_oSocket.m_iSocketHandle, &m_oWriteSet) )
	{
		int iSendResult = m_oSocket.Send();
		LOG_MESSAGE(3, "Has sent [%d] bytes to socket", iSendResult);
	}

	// Check if socket has a pending error
	if( FD_ISSET(m_oSocket.m_iSocketHandle, &m_oErrorSet) )
	{
		LOG_MESSAGE( 1, "Socket has a pending error: %s", strerror(errno) );

		m_oSocket.m_bClosed = true;
	}
}

void CProtocolHandler::Extract()
{
	tPacket* pPacket = NULL;
	do
	{
		if( m_oSocket.m_abtInBuffer.empty() ) break;

		pPacket = new tPacket();
		if( pPacket->Extract(m_oSocket.m_abtInBuffer) )
		{
			m_aInPacketBuffer.push_back(pPacket);
		}
		else
		{
			// TODO: add error validation VS normal termination

			delete pPacket;

			// m_oSocket.m_bClosed = true;

			// Got not enough data to extract packet

			// TODO: handle the case of a packet larger than inbound buffer

			break;
		}

	} while(pPacket);
}

void CProtocolHandler::Pack()
{
	while( !m_aOutPacketBuffer.empty() )
	{
		int iOldBufferSize = m_oSocket.m_abtOutBuffer.size();
		tPacket* pPacket = m_aOutPacketBuffer.front();
		const vector<BYTE> aPacket = pPacket->Pack();

		if(aPacket.size() > SOCKET_OUT_BUFFER_MAX_SIZE)
		{
			LOG_MESSAGE( 1, "Out byte buffer overflow: Packet size(%d) exceeds max buffer size: Erasing message", aPacket.size() );
			delete pPacket;
			m_aOutPacketBuffer.pop_front();

			// TODO: handle the case of a packet larger than outgoing buffer
		}

		if(iOldBufferSize + aPacket.size() > SOCKET_OUT_BUFFER_MAX_SIZE)
		{
			LOG_MESSAGE(3, "Socket: Out byte buffer overflow: Sending existing data first");
			break;
		}
		else
		{
			LOG_MESSAGE( 3, "Packed message for socket: %s", pPacket->ToString().c_str() );
			delete pPacket;
			m_aOutPacketBuffer.pop_front();

			m_oSocket.m_abtOutBuffer.insert( m_oSocket.m_abtOutBuffer.end(), aPacket.begin(), aPacket.end() );
		}
	}
}

void CProtocolHandler::AddOutgoingPacket(tPacket* pPacket)
{
	assert(pPacket);

	m_aOutPacketBuffer.push_back(pPacket);
}

tPacket* CProtocolHandler::GrabPacket()
{
	if( m_aInPacketBuffer.empty() )
		return NULL;

	tPacket* pPacket = m_aInPacketBuffer.front();
	m_aInPacketBuffer.pop_front();
	return pPacket;
}




void CServerProtocolHandler::Execute()
{
	const tPacket* pPacket = GrabPacket();
	if(pPacket)
	{
		LOG_MESSAGE( 3, "Processing packet \"%s\"", pPacket->ToString().c_str() );

		switch(pPacket->m_eType)
		{

		case tPacket::tType_GetSystemInfo:
		{
			// TODO: Handle GetSystemInfo
		}
			break;

		case tPacket::tType_Startup:
		{
			// TODO: Handle Startup

			LOG_MESSAGE(3, "Server got tType_Startup!");
		}
			break;

		case tPacket::tType_Exec:
		{
			// TODO: Handle Exec
		}
			break;

		case tPacket::tType_Read:
		{
			// TODO: Handle Read
		}
			break;

		case tPacket::tType_Shutdown:
		{
			// TODO: Handle Shutdown
		}
			break;

		default:
			LOG_MESSAGE(1, "Got unknown packet type");
			break;

		}
	}
}
