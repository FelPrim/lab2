# Компилятор (должен поддерживать C23)
CC = gcc

# Режим (DEBUG / RELEASE)
BUILD_MODE ?= DEBUG

EXE = lab2
TST = test

INCLUDE_DIR = .
EXE_DIR = .
TST_DIR = .
TMP_DIR = tmp
SRC_DIR = src

MAIN := $(SRC_DIR)/main.c
TEST := $(SRC_DIR)/test.c
ALL_C_FILES := $(wildcard $(SRC_DIR)/*.c)
SRC := $(filter-out $(MAIN) $(TEST), $(ALL_C_FILES))

HEADERS = $(wildcard $(SRC_DIR)/*.h)

ALLWAYS_ARE_CFLAGS = -std=c23 -Wall -Wextra -I$(INCLUDE_DIR)
CFLAGS_DEBUG = $(ALLWAYS_ARE_CFLAGS) -O0 -g
RELEASE_FLAGS = $(ALLWAYS_ARE_CFLAGS) -Ofast -flto -march=native -mtune=native -fipa-pta
CFLAGS_PGO_GEN = $(RELEASE_FLAGS) -fprofile-generate
CFLAGS_PGO_USE = $(RELEASE_FLAGS) -fprofile-use
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
	$(CC) $(CFLAGS_DEBUG) $^ -o $@ $(LDFLAGS)
else
	$(CC) $(CFLAGS_PGO_GEN) $^ -o $@ $(LDFLAGS)
	$(EXE_DIR)/$(EXE)
	$(CC) $(CFLAGS_PGO_USE) $^ -o $@ $(LDFLAGS)
endif

$(TST_DIR)/$(TST): $(TEST) $(SRC)
	$(CC) $(CFLAGS_DEBUG) $^ -o $@ $(LDFLAGS)

run_test: $(TST_DIR)/$(TST)
	@$(TST_DIR)/$(TST)

clean:
	rm -f $(EXE_DIR)/$(EXE) $(TST_DIR)/$(TST) $(TMP_DIR)/*
