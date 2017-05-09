#ifndef ATHENA__LOG_H
#define ATHENA__LOG_H

#ifdef LOG_DEBUG
#define debug(frame, expr)   _log(frame, "debug") << expr;
#define info(frame, expr)    _log(frame, "info") << expr;
#define warning(frame, expr) _log(frame, "warning") << expr;
#define error(frame, expr)   _log(frame, "error") << expr;
#else
#ifdef LOG_INFO
#define debug(frame, expr)   ;
#define info(frame, expr)    _log(frame, "info") << expr;
#define warning(frame, expr) _log(frame, "warning") << expr;
#define error(frame, expr)   _log(frame, "error") << expr;
#else
#ifdef LOG_WARNING
#define debug(frame, expr)   ;
#define info(frame, expr)    ;
#define warning(frame, expr) _log(frame, "warning") << expr;
#define error(frame, expr)   _log(frame, "error") << expr;
#else
#ifdef LOG_ERROR
#define debug(frame, expr)   ;
#define info(frame, expr)    ;
#define warning(frame, expr) ;
#define error(frame, expr)   _log(frame, "error") << expr;
#else
#define debug(frame, expr)   ;
#define info(frame, expr)    ;
#define warning(frame, expr) ;
#define error(frame, expr)   ;
#endif
#endif
#endif
#endif


#include <iostream>


std::ostream& _log(const char *frame, const char *level);

#endif
