#ifndef _PROTOCOL_HANDLER_H_
#define _PROTOCOL_HANDLER_H_

#include "DataTypes.h"
#include "Socket.h"
#include "Packet.h"

#define IN_PACKET_BUFFER_MAX_SIZE    5
#define OUT_PACKET_BUFFER_MAX_SIZE    5

class CProtocolHandler
{
public:
    CProtocolHandler(int iSocketFD);
    virtual ~CProtocolHandler();
    
    void Write(tPacket *pPacket);
    tPacket Read();
    
private:
    tSocket m_oSocket;
};

#endif // _PROTOCOL_HANDLER_H_
