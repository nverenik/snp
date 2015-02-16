#include "ProtocolHandler.h"
#include "RenameMe.h"

CProtocolHandler::CProtocolHandler(int iSocketFD)
{
    m_oSocket.m_iSocketHandle = iSocketFD;
}

CProtocolHandler::~CProtocolHandler()
{
    //assert(m_oSocket.m_bClosed == true);
    CloseSocket(m_oSocket.m_iSocketHandle);
}

void CProtocolHandler::Write(tPacket *pPacket)
{
    assert(!m_oSocket.m_bClosed && m_oSocket.m_iSocketHandle != -1);

    const std::vector<BYTE> aPacket = pPacket->Pack();
    if (aPacket.size() > SOCKET_OUT_BUFFER_MAX_SIZE)
        LOG_MESSAGE(1, "Out byte buffer overflow: Packet size(%d) exceeds max buffer size: Erasing message", aPacket.size());

    LOG_MESSAGE(5, "Packed message for socket: %s", pPacket->ToString().c_str());
    m_oSocket.m_abtOutBuffer.insert(m_oSocket.m_abtOutBuffer.end(), aPacket.begin(), aPacket.end());

    // Send outgoing data
    int iSendResult = m_oSocket.Send();
    LOG_MESSAGE(5, "Has sent [%d] bytes to socket", iSendResult);
}

tPacket CProtocolHandler::Read()
{
    assert(!m_oSocket.m_bClosed && m_oSocket.m_iSocketHandle != -1);
    int iReadResult = m_oSocket.Read();
    if(iReadResult > 0)
    {
        LOG_MESSAGE(5, "Has read [%d] bytes from socket", iReadResult);
    }
    else if(iReadResult <= 0) // Close socket on error | gentle shutdown
    {
        m_oSocket.m_bClosed = true;
    }

    tPacket oPacket;
    if (!oPacket.Extract(m_oSocket.m_abtInBuffer))
        LOG_MESSAGE(1, "Received invalid packet");

    return oPacket;
}
