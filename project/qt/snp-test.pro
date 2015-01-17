TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
QMAKE_CXXFLAGS += -std=c++11

SNPPATH = ../..
SNPLIB = snp

INCLUDEPATH += \
    $$SNPPATH/include

SOURCES += \
    ../../source/main.cpp

ConfigCUDA {
    SNPLIB = $$join(SNPLIB,,,.cuda)
}

ConfigRocksDB {
    SNPLIB = $$join(SNPLIB,,,.rocksdb)
}

CONFIG(debug, debug|release) {
    SNPLIB = $$join(SNPLIB,,,.debug)
}

LIBS += $$quote(-L$$SNPPATH/prebuilt) -l$$SNPLIB
