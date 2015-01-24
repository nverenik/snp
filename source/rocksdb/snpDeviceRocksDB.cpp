#include <snp/snpDevice.h>

#include <cstdio>
#include <iostream>

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <iostream>

#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>

extern "C" const int32 kCellNotFound = -1;
static int32 s_iCellIndex = kCellNotFound;

inline static const std::string bitfield2string(const std::vector<uint32> &value)
{
    std::ostringstream result;
    if (!value.empty())
    {
        std::copy(value.begin(), value.end() - 1, std::ostream_iterator<uint32>(result, " "));
        result << value.back();
    }

    return result.str();
}

inline static const std::vector<uint32> string2bitfield(const std::string value)
{
    std::vector<uint32> result;
    std::istringstream values(value);

    std::copy(std::istream_iterator<uint32>(values), std::istream_iterator<uint32>(), std::back_inserter(result));
    return result;
}

inline static bool validateCell(std::vector<uint32> &bitfield, const std::vector<uint32> mask, const std::vector<uint32> data)
{
    for (uint32 index = 0; index < bitfield.size(); index++)
    {
        if (snpCompareBits(bitfield[index], mask[index], data[index]) != true)
        {
            return false;
        }
    }
    return true;
}

inline static void performOperation(std::vector<uint32> &bitfield, const std::vector<uint32> mask, const std::vector<uint32> data, snp::snpOperation operation)
{
    uint32 bData[bitfield.size()];
    uint32 size = bitfield.size();

    std::copy(bitfield.begin(), bitfield.end(), bData);

    switch(operation)
    {
        case snp::snpAssign:	snpUpdateBits(snpUpdateBitsASSIGN,	bData[index], mask[index], data[index], bitfield.size());	break;
        case snp::snpNot:		snpUpdateBits(snpUpdateBitsNOT,		bData[index], mask[index], data[index], bitfield.size());	break;
        case snp::snpAnd:		snpUpdateBits(snpUpdateBitsAND,		bData[index], mask[index], data[index], bitfield.size());	break;
        case snp::snpOr:		snpUpdateBits(snpUpdateBitsOR,		bData[index], mask[index], data[index], bitfield.size());	break;
        default: break;
    }

    bitfield.assign(bData, bData + size);
}

NS_SNP_BEGIN

using namespace rocksdb;
using namespace std;

static const std::string kDatabasePath = "/tmp/rocksdb_table";

static DB *s_database = nullptr;

bool snpDeviceImpl::systemInfo()
{
	printf("snpDeviceImpl::systemInfo() - not implemented for RocksDB backend\n");
	return true;
}

bool snpDeviceImpl::init(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
    // setup device settings
    m_cellSize = cellSize;
    m_cellsPerPU = cellsPerPU;
    m_numberOfPU = numberOfPU;
    s_iCellIndex = kCellNotFound;

    // initialize database
    Options options;

    // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
    options.IncreaseParallelism();
    options.OptimizeLevelStyleCompaction();

    // create the DB if it's not already present
    options.create_if_missing = true;

    // initialize database
    Status dbStatus = DB::Open(options, kDatabasePath + std::to_string(m_cellSize), &s_database);
    if (dbStatus.ok())
    {
        // fillup database with default values
        rocksdb::WriteBatch initBatch; Status dbStatus;

        std::vector<uint32> defaultVector;
        for(uint16 index = 0; index < m_cellSize; index++)
        {
            defaultVector.push_back(0);
        }

        std::string defaultValue = bitfield2string(defaultVector);
        for (uint32 index = 0; index < getCellsPerPU() * getNumberOfPU(); index++)
        {
            std::string key = std::to_string(index);
            std::string value;

            dbStatus = s_database->Get(rocksdb::ReadOptions(), key, &value);
            if (!dbStatus.ok() || string2bitfield(value).size() < m_cellSize)
            {
                initBatch.Put(key, defaultValue);
            }
        }

        dbStatus = s_database->Write(rocksdb::WriteOptions(), &initBatch);
    }
    return dbStatus.ok();
}

void snpDeviceImpl::deinit()
{
    delete s_database;
    s_database = nullptr;
}

bool snpDeviceImpl::exec(bool singleCell, snpOperation operation, const uint32 * const instruction)
{
    // get instruction data
    const uint32 * const addressMask = instruction + 0 * m_cellSize;
    const uint32 * const addressData = instruction + 1 * m_cellSize;
    const uint32 * const   valueMask = instruction + 2 * m_cellSize;
    const uint32 * const   valueData = instruction + 3 * m_cellSize;

    // find cells from instuction
    std::vector<uint32> mask; mask.assign(addressMask, addressMask + m_cellSize);
    std::vector<uint32> data; data.assign(addressData, addressData + m_cellSize);

    std::vector<std::pair<string, string>> validatedPairs;

    rocksdb::Iterator *it = s_database->NewIterator(rocksdb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next())
    {
        std::vector<uint32> bitfield = string2bitfield(it->value().ToString());
        if (validateCell(bitfield, mask, data))
        {
            validatedPairs.push_back(std::make_pair(it->key().ToString(), it->value().ToString()));
            if (singleCell)
                break;
        }
    }

    // apply operation to validated cells
    rocksdb::WriteBatch batchOperation;

    mask.assign(valueMask, valueMask + m_cellSize);
    data.assign(valueData, valueData + m_cellSize);

    for (auto it = validatedPairs.begin(); it != validatedPairs.end(); ++it)
    {
        std::vector<uint32> bitfield = string2bitfield(it->second);
        performOperation(bitfield, mask, data, operation);
        batchOperation.Put(it->first, bitfield2string(bitfield));
    }

    Status dbStatus = s_database->Write(rocksdb::WriteOptions(), &batchOperation);

    // save first validated cell into buffer
    s_iCellIndex = (validatedPairs.size() > 0) ? std::stoul(validatedPairs.at(0).first) : kCellNotFound;
    return (dbStatus.ok() && (validatedPairs.size() > 0));
}

bool snpDeviceImpl::read(uint32 *bitfield)
{
    std::string value;
    Status dbStatus = s_database->Get(rocksdb::ReadOptions(), std::to_string(s_iCellIndex), &value);
    if (dbStatus.ok())
    {
        vector<uint32> bfield = string2bitfield(value);
        std::copy(bfield.begin(), bfield.end(), bitfield);
        return true;
    }
	return false;
}

void snpDeviceImpl::dump()
{
    rocksdb::Iterator* it = s_database->NewIterator(rocksdb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next())
    {
        std::cout << it->value().ToString() << std::endl;
    }
    assert(it->status().ok());
}

NS_SNP_END
