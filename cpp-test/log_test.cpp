#include "log_test.h"
#include "_log.h"

#include <gtest/gtest.h>
#include <iostream>


using namespace std;


TEST(log_debug, does_not_crash) {
  debug(__func__, "test" << endl);
}

TEST(log_info, does_not_crash) {
  info(__func__, "test" << endl);
}

TEST(log_warning, does_not_crash) {
  warning(__func__, "test" << endl);
}

TEST(log_error, does_not_crash) {
  error(__func__, "test" << endl);
}
