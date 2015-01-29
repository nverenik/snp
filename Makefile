# Location of the CUDA Toolkit
CUDA_PATH       ?= /opt/cuda
MPI_PATH		?= /opt/intel/impi/4.1.0

OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
	XCODE_GE_5 = $(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5)
endif

# Take command line flags that override any of these settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
ifeq ($(ARMv7),1)
	OS_SIZE = 32
	OS_ARCH = armv7l
endif

# Common binaries
ifneq ($(DARWIN),)
  ifeq ($(XCODE_GE_5),1)
    GCC ?= clang
  else
    GCC ?= gcc
  endif
else
  GCC ?= gcc
endif
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

# Common path
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_COMMON_PATH?= $(CUDA_PATH)/common
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

ifeq ($(OS_SIZE),32)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
endif

# mpi
ifeq ($(OS_SIZE),32)
  MPI_INCLUDE	?= $(MPI_PATH)/include
  MPI_LIB_PATH	?= $(MPI_PATH)/lib

  # LD_FLAGS += -L$(MPI_LIB_PATH) -lmpi
  # cxx.lib;fmpich2.lib;fmpich2g.lib;fmpich2s.lib;mpe.lib;mpi.lib
else
  MPI_INCLUDE	?= $(MPI_PATH)/include64
  MPI_LIB_PATH	?= $(MPI_PATH)/lib64

  # LD_FLAGS += -L$(MPI_LIB_PATH) -lmpi
  # cxx.lib;fmpich2.lib;fmpich2g.lib;irlog2rlog.lib;mpe.lib;mpi.lib;rlog.lib;TraceInput.lib
endif

# internal flags
CU_FLAGS := -m${OS_SIZE} --ptxas-options=-v -keep -Xcompiler -fPIC
CC_FLAGS := -fPIC -std=c++0x -lstdc++ -O3
LD_FLAGS := -L$(CUDA_LIB_PATH) -lcudart
LD_FLAGS += -L$(MPI_LIB_PATH) -lmpi

# OS-specific build flags
ifneq ($(DARWIN),)
  CC_FLAGS += -arch $(OS_ARCH)
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),gnueabi)
      CC_FLAGS += -mfloat-abi=softfp
    else
      # default to gnueabihf
      override abi := gnueabihf
#      LD_FLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
      CC_FLAGS += -mfloat-abi=hard
    endif
  endif
endif

ifeq ($(OS_ARCH),armv7l)
  CU_FLAGS += -target-cpu-arch ARM
  ifneq ($(TARGET_FS),)
    CC_FLAGS += --sysroot=$(TARGET_FS)
	
    LD_FLAGS += --sysroot=$(TARGET_FS)
    LD_FLAGS += -rpath-link=$(TARGET_FS)/lib
    LD_FLAGS += -rpath-link=$(TARGET_FS)/usr/lib
    LD_FLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
	
	OPEN_GL_PATH ?= -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
  endif
endif

# Common includes and paths for CUDA
INCLUDES := -Iinclude/ \
			-Iexternal/tclap-1.2.1/include/ \
			-O3 \
			-I$(CUDA_INC_PATH) \
			-I$(MPI_INCLUDE)

# CUDA code generation flags
GENCODE_SM20 := -gencode arch=compute_20,code=sm_20
GENCODE_SM30 := -gencode arch=compute_30,code=sm_30
#GENCODE_SM32 := -gencode arch=compute_32,code=sm_32
#GENCODE_SM35 := -gencode arch=compute_35,code=sm_35
#GENCODE_SM50 := -gencode arch=compute_50,code=sm_50
#GENCODE_SMXX := -gencode arch=compute_50,code=compute_50
ifeq ($(OS_ARCH),armv7l)
  GENCODE_FLAGS ?= $(GENCODE_SM32)
else
  GENCODE_FLAGS ?= $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)
endif

CC_FLAGS += $(INCLUDES)
CU_FLAGS += $(INCLUDES) $(GENCODE_FLAGS)

# CC_FLAGS += -DSNP_TARGET_CUDA

SNPWORKER_ROOT = source/mpicuda/snpworker
SNPWORKER_OBJ = kernel.o main.o
# SNP_OBJECT = snpDeviceRocksDB.o snpDeviceCUDA.o snpDevice.o kernel.o main.o

all: $(SNPWORKER_OBJ)
	$(GCC) -o snpworker $(SNPWORKER_OBJ) $(LD_FLAGS) -lstdc++ -lm
	#-L$(CUDA_LIB_PATH) -L$(MPI_LIB_PATH) -lm
	make clean

clean:
	rm -f *.o *.cudafe* *cpp*i *cubin *cu.cpp *fatbin* *linkinfo *.ptx *hashinfo *hash *.module_id

# snpDeviceRocksDB.o: ./snpDeviceRocksDB.cpp
# 	$(GCC) $(CC_FLAGS) -c ./snpDeviceRocksDB.cpp

# snpDeviceCUDA.o: ./snpDeviceCUDA.cpp
# 	$(GCC) $(CC_FLAGS) -c ./snpDeviceCUDA.cpp

# snpDevice.o: ./snpDevice.cpp
# 	$(GCC) $(CC_FLAGS) -c ./snpDevice.cpp

main.o: $(SNPWORKER_ROOT)/main.cpp
	$(GCC) $(CC_FLAGS) -c $(SNPWORKER_ROOT)/main.cpp

kernel.o: $(SNPWORKER_ROOT)/kernel.cu
	$(NVCC) $(CU_FLAGS) -c $(SNPWORKER_ROOT)/kernel.cu
	