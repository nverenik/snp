#include "Packet.h"

bool tPacket::Extract(std::vector<BYTE>& raBuffer)
{
    static const unsigned int iMinPacketSize = sizeof(WORD) + sizeof(DWORD) + sizeof(DWORD) + 0 + sizeof(DWORD);

    // 1: Extract starter
    if( raBuffer.size() < iMinPacketSize )
        return false;

    BYTE* pBuffer = &( raBuffer.front() );

    WORD wTestStarter = ExtractWord(pBuffer);
    if(wTestStarter != PACKET_STARTER)
    {
        while( raBuffer.size() >= sizeof(WORD) )
        {
            wTestStarter = ExtractWord(pBuffer);
            if(wTestStarter == PACKET_STARTER)
            {
                break;
            }
            pBuffer = pop_front( raBuffer, sizeof(WORD) );
        }

        if(wTestStarter != PACKET_STARTER)
            return false;
    }

    if( raBuffer.size() < iMinPacketSize )
        return false;

    // 2: Extract packet type
    DWORD dwPacketType = ExtractDword( &*(raBuffer.begin() + sizeof(WORD)) );
    if(dwPacketType >= tType_NUMTYPES)
    {
        LOG_MESSAGE(1, "Got a package with wrong type (%d)!", dwPacketType);

        pop_front( raBuffer, sizeof(WORD) );

        return false;
    }

    // 3: Extract data size
    DWORD dwDataSize = ExtractDword( &*(raBuffer.begin() + sizeof(WORD) + sizeof(DWORD)) );
    if(dwDataSize > PACKET_MAX_SIZE)
    {
        LOG_MESSAGE(1, "Got a package with wrong data size (%d)!", dwDataSize);

        pop_front( raBuffer, sizeof(WORD) );

        return false;
    }

    // 4: Extract data block
    if( raBuffer.size() < iMinPacketSize + dwDataSize )
        return false;

    // 5: Extract & check Crc32
    /*DWORD dwCrc32 = ExtractDword(sizeof(WORD) + sizeof(DWORD) + sizeof(DWORD) + dwDataSize);
    DWORD dwCrc32Check = CRC::ArrayCrc32(pBuffer + sizeof(WORD) + sizeof(DWORD), dwDataSize - sizeof(DWORD));
    if(dwCrc32 != dwCrc32Check)
    {
        LOG_MESSAGE(1, "Got a package with wrong CRC!", dwDataSize);

        pop_front( raBuffer, sizeof(WORD) );

        return false;
    }*/

    m_eType = (tType)dwPacketType;
    memcpy(m_oData.m_oU.m_aByteData, &*(raBuffer.begin() + sizeof(WORD) + sizeof(DWORD) + sizeof(DWORD)), dwDataSize);

    pop_front(raBuffer, iMinPacketSize + dwDataSize);

    return true;
}

std::vector<BYTE> tPacket::Pack()
{
    std::vector<BYTE> aPacket;

    DWORD dwDataSize = 0;
    switch(m_eType)
    {
    case tType_Startup: dwDataSize = sizeof(m_oData.m_oU.m_oDataStartup); break;
    case tType_Exec: dwDataSize = sizeof(m_oData.m_oU.m_oDataExec); break;
    case tType_Read: dwDataSize = sizeof(m_oData.m_oU.m_oDataRead); break;
    default: break;
    }

    DWORD dwCrc32 = 0;

    AppendWord(aPacket, PACKET_STARTER);
    AppendDword(aPacket, (DWORD)m_eType);
    AppendDword(aPacket, dwDataSize);
    aPacket.insert( aPacket.end(), m_oData.m_oU.m_aByteData, m_oData.m_oU.m_aByteData + dwDataSize );
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
    if( dwSize > raBuffer.size() )
    {
        assert( dwSize <= raBuffer.size() );
        dwSize = raBuffer.size();
    }

    raBuffer.erase( raBuffer.begin(), raBuffer.begin() + dwSize );

    return &(raBuffer.front());
}
