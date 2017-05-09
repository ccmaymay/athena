#!/usr/bin/env python


from athena.twitter import extract_redis_tweet_text
from athena.util import parse_basic_args
from athena.util import infer_redis_addr
import logging


def main():
    (redis_db, _) = parse_basic_args(
        'load twitter comms from redis, parse into light-weight dicts, and'
        'filter')
    (host, port) = infer_redis_addr(redis_db)

    input_key = 'twitter/comms'
    num_processes = 16
    batch_size = 10000
    sleep_interval = 0.1
    max_batch_lag = 100
    context_size = 5

    total = 0
    for d in extract_redis_tweet_text(host, port, input_key,
                                      num_processes=num_processes,
                                      batch_size=batch_size,
                                      sleep_interval=sleep_interval,
                                      context_size=context_size,
                                      max_batch_lag=max_batch_lag):
        total += 1
        if total % 10000 == 0:
            logging.info(u'tweet %d' % total)


if __name__ == '__main__':
    main()
