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




//    CProtocolHandler();
//    virtual ~CProtocolHandler();
//    void Tick();
//    void InitSocket(int iSocketFD);
//    void AddOutgoingPacket(tPacket* pPacket);
//
//
//private:
//    
//    std::list<tPacket*> m_aInPacketBuffer;
//    std::list<tPacket*> m_aOutPacketBuffer;
//
//    fd_set m_oReadSet;
//    fd_set m_oWriteSet;
//    fd_set m_oErrorSet;
//
//    void Select();
//    void ReadWrite();
//    void Extract();
//    void Pack();
//
//    tPacket* GrabPacket();
};

#endif // _PROTOCOL_HANDLER_H_
