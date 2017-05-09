#!/usr/bin/env python


from athena.util import parse_basic_args
from athena.scion import load_accumulo_sequential, DEFAULT_TWITTER_ANALYTICS


def main():
    (redis_db, _) = parse_basic_args(
        'load twitter comms into redis, chronologically')

    corpus = 'twitter'
    output_key = 'twitter/comms'

    load_accumulo_sequential(
        redis_db, corpus, output_key, irresponsible_gc=True, batch_size=10000,
        key_limit=1000000, poll_interval=500,
        analytics=DEFAULT_TWITTER_ANALYTICS + ('Twitter LID-1',),
        begin_range=682712843158958080,
        end_range=990000000000000000)


if __name__ == '__main__':
    main()
