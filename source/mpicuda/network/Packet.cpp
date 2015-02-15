#include "Packet.h"

bool tPacket::Extract(std::vector<BYTE> &raBuffer)
{
    static const unsigned int iMinPacketSize
        = sizeof(WORD)      // starter
        + sizeof(DWORD)     // packet type
        + sizeof(DWORD)     // size of data block
        + 0                 // data block
        + sizeof(DWORD)     // size of dynamic data block
        + 0                 // dynamic data block
        + sizeof(DWORD);    // crc32

    // 1: Extract starter
    if (raBuffer.size() < iMinPacketSize)
        return false;

    // pointer to the current packet field
    BYTE *pBuffer = &raBuffer.front();

    WORD wTestStarter = ExtractWord(pBuffer);
    if (wTestStarter != PACKET_STARTER)
    {
        // skip data until starter is found
        while (raBuffer.size() >= sizeof(WORD))
        {
            wTestStarter = ExtractWord(pBuffer);
            if(wTestStarter == PACKET_STARTER)
                break;
            pBuffer = pop_front(raBuffer, sizeof(WORD));
        }

        if (wTestStarter != PACKET_STARTER)
            return false;
    }

    if (raBuffer.size() < iMinPacketSize)
        return false;

    // shift buffer pointer by the starter size
    pBuffer += sizeof(WORD);

    // 2: Extract packet type
    DWORD dwPacketType = ExtractDword(pBuffer);
    if(dwPacketType >= tType_NUMTYPES)
    {
        LOG_MESSAGE(1, "Got a package with wrong type (%d)!", dwPacketType);
        pop_front(raBuffer, sizeof(DWORD));
        return false;
    }
    pBuffer += sizeof(DWORD);

    // 3: Extract data size
    DWORD dwDataSize = ExtractDword(pBuffer);
    if (dwDataSize > PACKET_MAX_SIZE)
    {
        LOG_MESSAGE(1, "Got a package with wrong data size (%d)!", dwDataSize);
        pop_front(raBuffer, sizeof(DWORD));
        return false;
    }
    pBuffer += sizeof(DWORD);

    // 4: Extract data block
    if (raBuffer.size() < iMinPacketSize + dwDataSize)
        return false;

    m_eType = (tType)dwPacketType;
    memcpy(m_oData._raw, pBuffer, dwDataSize);
    pBuffer += dwDataSize;
    
    // 5. Extract dynamic data size
    DWORD dwDynamicDataSize = ExtractDword(pBuffer);
    if (dwDataSize > PACKET_MAX_SIZE)
    {
        LOG_MESSAGE(1, "Got a package with wrong data size (%d)!", dwDataSize);
        pop_front(raBuffer, sizeof(DWORD));
        return false;
    }
    pBuffer += sizeof(DWORD);

    // 6. Extract dynamic data block
    if (raBuffer.size() < iMinPacketSize + dwDataSize + dwDynamicDataSize)
        return false;

    m_oDynamicData.resize(dwDynamicDataSize);
    if (dwDynamicDataSize > 0)
    {
        memcpy(&m_oDynamicData.front(), pBuffer, dwDynamicDataSize);
        pBuffer += dwDynamicDataSize;
    }   

    // 7. Extract & check Crc32
    /*DWORD dwCrc32 = ExtractDword(sizeof(WORD) + sizeof(DWORD) + sizeof(DWORD) + dwDataSize);
    DWORD dwCrc32Check = CRC::ArrayCrc32(pBuffer + sizeof(WORD) + sizeof(DWORD), dwDataSize - sizeof(DWORD));
    if(dwCrc32 != dwCrc32Check)
    {
        LOG_MESSAGE(1, "Got a package with wrong CRC!", dwDataSize);
        pop_front( raBuffer, sizeof(WORD) );
        return false;
    }*/

    pop_front(raBuffer, iMinPacketSize + dwDataSize + dwDynamicDataSize);
    return true;
}

std::vector<BYTE> tPacket::Pack() const
{
    std::vector<BYTE> aPacket;

    DWORD dwDataSize = 0;
    switch(m_eType)
    {
        case tType_RequestStartup:      dwDataSize = sizeof(m_oData.asRequestStartup); break;
        case tType_RequestShutdown:     dwDataSize = sizeof(m_oData.asRequestShutdown); break;
        case tType_RequestExec:         dwDataSize = sizeof(m_oData.asRequestExec); break;
        case tType_RequestRead:         dwDataSize = sizeof(m_oData.asRequestRead); break;

        case tType_ResponseStartup:     dwDataSize = sizeof(m_oData.asResponseStartup); break;
        case tType_ResponseShutdown:    dwDataSize = sizeof(m_oData.asResponseShutdown); break;
        case tType_ResponseExec:        dwDataSize = sizeof(m_oData.asResponseExec); break;
        case tType_ResponseRead:        dwDataSize = sizeof(m_oData.asResponseRead); break;

        default: break;
    }

    DWORD dwCrc32 = 0;

    AppendWord(aPacket, PACKET_STARTER);
    AppendDword(aPacket, (DWORD)m_eType);
    AppendDword(aPacket, dwDataSize);
    aPacket.insert(aPacket.end(), m_oData._raw, m_oData._raw + dwDataSize);
    AppendDword(aPacket, m_oDynamicData.size());
    aPacket.insert(aPacket.end(), m_oDynamicData.begin(), m_oDynamicData.end());
    AppendDword(aPacket, dwCrc32);

    return aPacket;
}

void tPacket::AppendByte(std::vector<BYTE>& raBuffer, BYTE btValue)
{
    raBuffer.push_back(btValue);
}

void tPacket::AppendWord(std::vector<BYTE>& raBuffer, WORD wValue)
{
    AppendByte( raBuffer, LOBYTE(wValue) );
    AppendByte( raBuffer, HIBYTE(wValue) );
}

void tPacket::AppendDword(std::vector<BYTE>& raBuffer, DWORD dwValue)
{
    AppendWord( raBuffer, LOWORD(dwValue) );
    AppendWord( raBuffer, HIWORD(dwValue) );
}

BYTE tPacket::ExtractByte(const BYTE *pByte)
{
    BYTE btByte = *pByte;
    return btByte;
}

WORD tPacket::ExtractWord(const BYTE *pWord)
{
    WORD wWord = MAKEWORD( ExtractByte(pWord), ExtractByte(pWord + 1) );
    return wWord;
}

DWORD tPacket::ExtractDword(const BYTE *pDword)
{
    DWORD dwDword = MAKEDWORD( ExtractWord(pDword), ExtractWord(pDword + 2) );
    return dwDword;
}

BYTE* tPacket::pop_front(std::vector<BYTE>& raBuffer, DWORD dwSize)
{
    if (dwSize > raBuffer.size())
    {
        assert(dwSize <= raBuffer.size());
        dwSize = raBuffer.size();
    }

    raBuffer.erase(raBuffer.begin(), raBuffer.begin() + dwSize);
    return (raBuffer.size() > 0) ? &(raBuffer.front()) : NULL;
}
