#ifndef _PACKET_H_
#define _PACKET_H_

#include <snp/snpMacros.h>
#include <snp/snpOperation.h>

#include "DataTypes.h"

#define PACKET_STARTER      0xFFFF
#define PACKET_MAX_SIZE     65530

struct tPacket
{
    // Packet structure: [wStarter][dwType][dwSize]<btData>[dwCRC]

    enum tType
    {
        tType_RequestStartup = 0,
        tType_RequestExec,
        tType_RequestRead,
        tType_RequestShutdown,

        tType_ResponseStartup,
        tType_ResponseExec,
        tType_ResponseRead,
        tType_ResponseShutdown,

        tType_NUMTYPES
    };

    struct tData
    {
        union
        {
            BYTE _raw[1024];    // TODO: Specify MAX possible data size here

            struct
            {
                uint32  m_uiCellSize;
                uint32  m_uiCellsPerPU;
                uint32  m_uiNumberOfPU;
            } asRequestStartup;

            struct
            {
                bool m_bResult;
            } asResponseStartup;

            struct
            {
            } asRequestShutdown;

            struct
            {
                bool m_bResult;
            } asResponseShutdown;

            struct
            {
                bool m_bSingleCell;
                snp::tOperation m_eOperation;
            } asRequestExec;

            struct
            {
                bool m_bResult;
            } asResponseExec;

            struct
            {
            } asRequestRead;

            struct
            {
                bool m_bResult;
            } asResponseRead;
        };
    };

    typedef std::vector<BYTE> tDynamicData;

    tType m_eType;
    tData m_oData;
    tDynamicData m_oDynamicData;

    bool Extract(std::vector<BYTE>& raBuffer);
    std::vector<BYTE> Pack() const;

    std::string ToString() const
    {
        switch(m_eType)
        {
            case tType_RequestStartup:      return "Request Startup";
            case tType_RequestExec:         return "Request Exec";
            case tType_RequestRead:         return "Request Read";
            case tType_RequestShutdown:     return "Request Shutdown";

            case tType_ResponseStartup:     return "Response Startup";
            case tType_ResponseExec:        return "Response Exec";
            case tType_ResponseRead:        return "Response Read";
            case tType_ResponseShutdown:    return "Response Shutdown";

            default: break;
        }
        return "Undefined";
    }

    static void AppendByte(std::vector<BYTE>& raBuffer, BYTE btValue);
    static void AppendWord(std::vector<BYTE>& raBuffer, WORD wValue);
    static void AppendDword(std::vector<BYTE>& raBuffer, DWORD dwValue);

    static BYTE ExtractByte(const BYTE* pByte);
    static WORD ExtractWord(const BYTE* pWord);
    static DWORD ExtractDword(const BYTE* pDword);

    static BYTE* pop_front(std::vector<BYTE>& raBuffer, DWORD dwSize);

};

#endif // _PACKET_H_
