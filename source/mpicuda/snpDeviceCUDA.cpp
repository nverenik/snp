#include <snp/snpDevice.h>
#include "snpCommand.h"

NS_SNP_BEGIN

bool snpDeviceImpl::init(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
	// TODO:
	// 1. read config from file (can be hardcoded for now) and run worker process on
	// the cluster host machine via ssh using parameters from the config
	// 2. connect to the worker process
	// 3. (optional) request and print info about cluster (can be used as a test)
	// 4. send 'startup' command to the worker
	// 5. return True if startup was succeeded
	return false;
}

void snpDeviceImpl::deinit()
{
	// TODO:
	// 1. send 'shutdown' command to the worker (cluster process will be stopped)
	// 2. cleanup connection variables
	// 3. (optional) download log files from the cluster to local machine
}

bool snpDeviceImpl::exec(bool singleCell, snpOperation operation, const uint32 * const instruction)
{
	// TODO:
	// 1. send 'exec' command to the worker
	// 2. pass the responce above
	return false;
}

bool snpDeviceImpl::read(uint32 *bitfield)
{
	// TODO:
	// 1. send 'read' command to the worker
	// 2. pass the responce above
	return false;
}

void snpDeviceImpl::dump()
{
	// nothing
}

NS_SNP_END
