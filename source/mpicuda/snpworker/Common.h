#ifndef __COMMON_H__
#define __COMMON_H__

// Error handling macros
#define MPI_CHECK(__call__) \
if ((__call__) != MPI_SUCCESS) \
{ \
    cerr << "MPI error calling \""#__call__"\"\n"; \
    MPI_Abort(MPI_COMM_WORLD, -1); \
}

#endif //__COMMON_H__