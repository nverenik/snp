QT -= core gui
TEMPLATE = lib
CONFIG += staticlib
QMAKE_CXXFLAGS += -std=c++11

TARGET = snp

ConfigCUDA {
    TARGET = $$join(TARGET,,,.cuda)
    DEFINES += SNP_TARGET_CUDA

    HEADERS += \
        ../../source/kernel.h

    SOURCES += \
        ../../source/kernel.cu
}

ConfigRocksDB {
    TARGET = $$join(TARGET,,,.rocksdb)
    DEFINES += SNP_TARGET_ROCKS_DB

    INCLUDEPATH += \
        ../../external/rocksdb-3.8/include

    LIBS += -L../../external/rocksdb-3.8/bin -lrocksdb
}

CONFIG(debug, debug|release) {
    TARGET = $$join(TARGET,,,.debug)
}

INCLUDEPATH += \
    ../../include \
    ../../source

HEADERS += \
    ../../include/snp/snp.h \
    ../../include/snp/snpDevice.h \
    ../../include/snp/snpErrorCode.h \
    ../../include/snp/snpMacros.h \
    ../../include/snp/snpOperation.h \
    ../../source/snpBackendConfig.h

SOURCES += \
    ../../source/snpDevice.cpp \
    ../../source/snpDeviceRocksDB.cpp \
    ../../source/snpDeviceCUDA.cpp

unix {
    target.path = /usr/lib
    INSTALLS += target
}
