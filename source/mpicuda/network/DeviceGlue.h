#ifndef __DEVICE_GLUE_H__
#define __DEVICE_GLUE_H__

#include <snp/snp.h>

class CProtocolHandler;

NS_SNP_BEGIN

namespace glue {

bool Startup(CProtocolHandler *pHandler, uint16 &uiCellSize, uint32 &uiCellsPerPU, uint32 &uiNumberOfPU);
bool Shutdown(CProtocolHandler *pHandler);
bool Exec(CProtocolHandler *pHandler, bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction, uint32 uiInstructionSize);
bool Read(CProtocolHandler *pHandler, uint32 *pBitfield);

}

NS_SNP_END

#endif //__DEVICE_GLUE_H__