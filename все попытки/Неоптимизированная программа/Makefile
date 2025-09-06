# Компилятор (должен поддерживать C++23)
CC = g++

# Режим (DEBUG / RELEASE)
BUILD_MODE ?=  DEBUG

EXE = lab2
TST = test

INCLUDE_DIR = .
EXE_DIR = .
TST_DIR = .
TMP_DIR = tmp
SRC_DIR = src

MAIN := $(SRC_DIR)/main.cpp
TEST := $(SRC_DIR)/test.cpp
ALL_CPP_FILES := $(wildcard $(SRC_DIR)/*.cpp)
SRC := $(filter-out $(MAIN) $(TEST), $(ALL_CPP_FILES))

HEADERS = $(wildcard $(SRC_DIR)/*.hpp)

ALLWAYS_ARE_CPPFLAGS = -std=c++23 -Wall -Wextra -I$(INCLUDE_DIR)
CPPFLAGS_DEBUG = $(ALLWAYS_ARE_CPPFLAGS) -O0 -g
RELEASE_FLAGS = $(ALLWAYS_ARE_CPPFLAGS) -Ofast -flto -march=native
CPPFLAGS_PGO_GEN = $(RELEASE_FLAGS) -fprofile-generate
CPPFLAGS_PGO_USE = $(RELEASE_FLAGS) -fprofile-use
LDFLAGS = -lmatio

.PHONY: all debug release run_test clean

all:
ifeq ($(BUILD_MODE), DEBUG)
	$(MAKE) debug
else
	$(MAKE) release
endif

debug: $(EXE_DIR)/$(EXE) $(TST_DIR)/$(TST)

$(EXE_DIR)/$(EXE): $(MAIN) $(SRC)
ifeq ($(BUILD_MODE), DEBUG)
	$(CC) $(CPPFLAGS_DEBUG) $^ -o $@ $(LDFLAGS)
else
	$(CC) $(CPPFLAGS_PGO_GEN) $^ -o $@ $(LDFLAGS)
	$(EXE_DIR)/$(EXE)
	$(CC) $(CPPFLAGS_PGO_USE) $^ -o $@ $(LDFLAGS)
endif

$(TST_DIR)/$(TST): $(TEST) $(SRC)
	$(CC) $(CPPFLAGS_DEBUG) $^ -o $@ $(LDFLAGS)

release: $(EXE_DIR)/$(EXE)

run_test: $(TST_DIR)/$(TST)
	$(TST_DIR)/$(TST)

clean:
	rm -f $(EXE_DIR)/$(EXE) $(TST_DIR)/$(TST) $(TMP_DIR)/*