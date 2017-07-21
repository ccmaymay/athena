#include "_log.h"
#include <ctime>
#include <cstring>
#include <sys/time.h>


using namespace std;


std::ostream& _log(const char *frame, const char *level) {
  struct timeval tv;
  char s[128];
  int written;
  int ret = gettimeofday(&tv, NULL);
  if (ret == 0) {
    /* this will often (but not always) be correct */
    ret = strftime(s, 128, "%Y-%m-%d %H:%M:%S", localtime(&(tv.tv_sec)));
    if (ret == 0) {
      strcpy(s, "[time error]");
    } else {
      written = ret;
      ret = snprintf(s + written, 128 - written, ".%03zu",
                     (size_t) (tv.tv_usec / 1000));
      if (ret < 0 || ret >= 128 - written) {
        strcpy(s, "[time error]");
      }
    }
  } else {
    strcpy(s, "[time error]");
  }
  return std::cerr <<
    s << ": " <<
    frame << ": " <<
    level << ": ";
}
