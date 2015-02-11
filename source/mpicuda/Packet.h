#ifndef _PACKET_H_
#define _PACKET_H_

#include "DataTypes.h"

#define PACKET_STARTER	0xFFFF
#define PACKET_MAX_SIZE 65530

struct tPacket
{
	// Packet structure: [wStarter][dwType][dwSize]<btData>[dwCRC]

	enum tType
	{
		//tType_GetSystemInfo = 0,
		tType_Startup = 0,
		tType_Exec,
		tType_Read,
		tType_Shutdown,

		tType_NUMTYPES
	};

	struct tData
	{
		union
		{
			BYTE m_aByteData[1024];	// TODO: Specify MAX possible data size here

			struct tDataStartup
			{
				int m_iGPUs;		// Total number of GPUs to use
				int m_iPEs;			// Total number of PEs to use
				int m_iCells;		// Number of cells per PE
				int m_iCellSize;	// Cell size in bytes
			} m_oDataStartup;

			struct tDataExec
			{
				int m_iTmp;			// TODO: specify valid data format here
			} m_oDataExec;

			struct tDataRead
			{
				int m_iTmp;			// TODO: specify valid data format here
			} m_oDataRead;
		} m_oU;
	};

	tType m_eType;
	tData m_oData;

	bool Extract(std::vector<BYTE>& raBuffer);
	std::vector<BYTE> Pack();

	std::string ToString() const { return "Packet Name"; }

	static void AppendByte(std::vector<BYTE>& raBuffer, BYTE btValue);
	static void AppendWord(std::vector<BYTE>& raBuffer, WORD wValue);
	static void AppendDword(std::vector<BYTE>& raBuffer, DWORD dwValue);

	static BYTE ExtractByte(const BYTE* pByte);
	static WORD ExtractWord(const BYTE* pWord);
	static DWORD ExtractDword(const BYTE* pDword);

	static BYTE* pop_front(std::vector<BYTE>& raBuffer, DWORD dwSize);

};

#endif // _PACKET_H_
