#include "Socket.h"
#include "RenameMe.h"

tSocket::tSocket()
{
    m_bClosed = false;
    m_iSocketId = -1;
}

int tSocket::Read()
{
    assert(m_bClosed == false);

    int iReadResult = 0;
    int iReadSize = min2( SENDING_MAX_SIZE, SOCKET_IN_BUFFER_MAX_SIZE - (int)m_abtInBuffer.size() );
    iReadResult = ReadFromSocket(m_iSocketHandle, m_aReadBuffer, iReadSize);

    if(iReadResult == -1)
    {
        LOG_MESSAGE( 1, "read error for socket: %s", strerror(errno) );
    }
    if(iReadResult > 0)
    {
        m_abtInBuffer.insert(m_abtInBuffer.end(), m_aReadBuffer, m_aReadBuffer + iReadResult);
    }

    return iReadResult;
}

int tSocket::Send()
{
    assert(m_bClosed == false);

    int iSendSize = min2( SENDING_MAX_SIZE, (int)m_abtOutBuffer.size() );
    int iSendResult = WriteToSocket(m_iSocketHandle, &*(m_abtOutBuffer.begin()), iSendSize);

    if(iSendResult == -1)
    {
        LOG_MESSAGE( 1, "write error for socket: %s", strerror(errno) );
    }
    else if(iSendSize != iSendResult)
    {
        LOG_MESSAGE(3, "write: sent [%d] of [%d] byte(s) to socket", iSendResult, iSendSize);
    }
    if(iSendResult > 0)
    {
        m_abtOutBuffer.erase(m_abtOutBuffer.begin(), m_abtOutBuffer.begin() + iSendResult);
    }

    return iSendResult;
}
