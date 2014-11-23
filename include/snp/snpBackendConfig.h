#ifndef __SNP_BACKEND_CONFIG_H__
#define __SNP_BACKEND_CONFIG_H__

// define all supported backend implementations
#define SNP_BACKEND_UNKNOWN		0
#define SNP_BACKEND_CUDA		1
#define SNP_BACKEND_ROCKS_DB	2

// by default target backend implementation is unknown
#define SNP_TARGET_BACKEND		SNP_BACKEND_UNKNOWN

// determine target backend
#ifdef SNP_TARGET_CUDA
	#undef	SNP_TARGET_BACKEND
    #define	SNP_TARGET_BACKEND	SNP_BACKEND_CUDA
#endif //SNP_TARGET_CUDA

#ifdef SNP_TARGET_ROCKS_DB
	#undef	SNP_TARGET_BACKEND
	#define	SNP_TARGET_BACKEND	SNP_BACKEND_ROCKS_DB
#endif //SNP_TARGET_ROCKS_DB

// check user set backend
#if ! SNP_TARGET_BACKEND
    #error "Cannot recognize the target backend, check preprocessor definitions."
#endif 

#endif //__SNP_BACKEND_CONFIG_H__
