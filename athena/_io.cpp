#include "_io.h"


#include <string>
#include <stdexcept>


using namespace std;


void stream_ready_or_throw(ios& stream) {
  if (! stream) {
    throw runtime_error(string("stream_ready_or_throw: stream not ready"));
  }
}
