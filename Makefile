AR := ar
CXXFLAGS += -std=gnu++11 -Wall -Werror -pedantic -I$(SRC_DIR) -fopenmp -DLOG_INFO
TEST_CXXFLAGS += $(CXXFLAGS)
LDFLAGS += -L$(LIB_BUILD_DIR) -fopenmp
TEST_LDFLAGS += $(LDFLAGS)
LIBS +=
TEST_LIBS += -lgmock -lgtest -lgtest_main $(LIBS)

ifdef LOG_DEBUG
	CXXFLAGS += -DLOG_DEBUG
endif

ifdef GCOV
	CXXFLAGS += -fprofile-arcs -ftest-coverage
endif

ifdef GPROF
	CXXFLAGS += -g -pg
	LDFLAGS += -pg
endif

ifdef CACHEGRIND
	CXXFLAGS += -g
endif

ifdef VTUNE
	CXXFLAGS += -g
endif

ifdef DEBUG
	CXXFLAGS += -O0 -DLOG_DEBUG
	ifndef GPROF
		ifndef CACHEGRIND
			ifndef VTUNE
				CXXFLAGS += -g3 -gdwarf-2
			endif
		endif
	endif
else
	CXXFLAGS += -O3 -march=native -funroll-loops
endif

ifeq ($(shell uname -s),Darwin)
	CXX := clang++
	CXX_LD := clang++
	CBLAS_FLAGS ?= -framework Accelerate
	LDFLAGS += -undefined dynamic_lookup -arch x86_64
	ifdef GCOV
		LIBS += --coverage
	endif
else
	CXX := g++
	CXX_LD := g++
	CBLAS_FLAGS ?= -lopenblas
	ifdef DEBUG
		CXXFLAGS += -fvar-tracking-assignments
	endif
	LIBS += -lpthread
	ifdef GCOV
		LIBS += -lgcov
	endif
endif

ifeq ($(HAVE_CBLAS),1)
	LIBS += $(CBLAS_FLAGS)
	CXXFLAGS += -DHAVE_CBLAS
endif

INSTALL_BASE_DIR := $(PREFIX)

INSTALL_BIN_DIR := $(INSTALL_BASE_DIR)/bin
INSTALL_LIB_DIR := $(INSTALL_BASE_DIR)/lib
INSTALL_INCLUDE_DIR := $(INSTALL_BASE_DIR)/include

BUILD_BASE_DIR := build

SRC_DIR := src

MAIN_BUILD_DIR := $(BUILD_BASE_DIR)/bin
MAIN_SOURCES := \
    $(SRC_DIR)/spacesaving-word2vec-train.cpp \
    $(SRC_DIR)/spacesaving-word2vec-print.cpp \
    $(SRC_DIR)/word2vec-train.cpp \
    $(SRC_DIR)/word2vec-print.cpp \
    $(SRC_DIR)/spacesaving-lm-train.cpp \
    $(SRC_DIR)/spacesaving-lm-print.cpp \
    $(SRC_DIR)/naive-lm-train.cpp \
    $(SRC_DIR)/naive-lm-print.cpp \
    $(SRC_DIR)/word2vec-vocab-to-naive-lm.cpp
MAIN_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(MAIN_BUILD_DIR)/%.o,$(MAIN_SOURCES))
MAIN_NAMES := $(MAIN_OBJECTS:.o=)

LIB_BUILD_DIR := $(BUILD_BASE_DIR)/lib
LIB_NAME := $(LIB_BUILD_DIR)/libathena.a
LIB_SOURCES := $(filter-out $(MAIN_SOURCES),$(wildcard $(SRC_DIR)/*.cpp))
LIB_HEADERS := $(wildcard $(SRC_DIR)/*.h)
LIB_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(LIB_BUILD_DIR)/%.o,$(LIB_SOURCES))

TEST_SRC_DIR := test
TEST_BUILD_DIR := $(BUILD_BASE_DIR)/test
TEST_SOURCES := $(wildcard $(TEST_SRC_DIR)/*_test.cpp)
TEST_OBJECTS := $(patsubst $(TEST_SRC_DIR)/%.cpp,$(TEST_BUILD_DIR)/%.o,$(TEST_SOURCES))
TEST_PROGRAM := $(TEST_BUILD_DIR)/run_tests

MOCK_SOURCES := $(wildcard $(TEST_SRC_DIR)/*_mock.cpp)
MOCK_OBJECTS := $(patsubst $(TEST_SRC_DIR)/%.cpp,$(TEST_BUILD_DIR)/%.o,$(MOCK_SOURCES))

ifdef GCOV
	TEST_POSTPROC := gcovr -r . -e '^'$(TEST_SRC_DIR)'/.*' --gcov-executable gcov
else
	TEST_POSTPROC :=
endif

.PHONY: test
test: $(TEST_PROGRAM)
	OMP_NUM_THREADS=1 ./$(TEST_PROGRAM)
	$(TEST_POSTPROC)

.PHONY: valgrind-test
valgrind-test: $(TEST_PROGRAM)
	OMP_NUM_THREADS=1 valgrind -v --error-exitcode=1 --track-origins=yes ./$(TEST_PROGRAM)
	$(TEST_POSTPROC)

.PHONY: main
main: $(MAIN_NAMES)

.PHONY: lib
lib: $(LIB_NAME)

$(LIB_NAME): $(LIB_OBJECTS)
	@mkdir -p $(@D)
	$(AR) rs $@ $^

$(MAIN_OBJECTS): $(MAIN_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $< -c

$(LIB_OBJECTS): $(LIB_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h $(LIB_HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $< -c

$(MOCK_OBJECTS) $(TEST_OBJECTS): $(TEST_BUILD_DIR)/%.o: $(TEST_SRC_DIR)/%.cpp $(TEST_SRC_DIR)/%.h
	@mkdir -p $(@D)
	$(CXX) $(TEST_CXXFLAGS) -o $@ $< -c

$(MAIN_NAMES): %: %.o $(LIB_OBJECTS)
	@mkdir -p $(@D)
	$(CXX_LD) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TEST_PROGRAM): $(MOCK_OBJECTS) $(TEST_OBJECTS) $(LIB_OBJECTS)
	@mkdir -p $(@D)
	$(CXX_LD) $(TEST_LDFLAGS) -o $@ $^ $(TEST_LIBS)

.PHONY: clean
clean:
	rm -rf $(BUILD_BASE_DIR)
