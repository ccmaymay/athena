import os
import logging
from subprocess import check_call
from athena.util import infer_redis_addr


DEFAULT_LOG4J2_JSON_PATH = '/export/common/max/public/log4j2.json'
DEFAULT_SCION_CONF_PATH = os.path.expanduser('~/.scion.conf')
DEFAULT_CLASSPATH = os.path.expanduser('~/scion/concrete/target/*')

DEFAULT_GIGAWORD_ANALYTICS = ('Section', 'Sentence', 'Stanford CoreNLP-1',
                              'Stanford CoreNLP PTB-1')
DEFAULT_TWITTER_ANALYTICS = ('Section', 'Sentence', 'TweetInfo',
                             'Tift TwitterTokenizer 4.10.0 Tweet Tags-1',
                             'Tift TwitterTokenizer 4.10.0-1')

LOADER_PACKAGE = 'edu.jhu.hlt.scion.concrete.redis'


def scion_java_args(java_xmx, irresponsible_gc, scion_conf_path,
                    log4j2_json_path, classpath):
    return [
        'java',
        '-Xmx' + str(java_xmx),
        '-XX:+UseG1GC' if irresponsible_gc else '-XX:+UseSerialGC',
        '-Dconfig.file=' + scion_conf_path,
        '-Dlog4j.configurationFile=' + log4j2_json_path,
        '-cp', classpath,
    ]


def log_check_call(args):
    logging.info(' '.join(u"'%s'" % arg for arg in args))
    try:
        # if classpath is a single directory, expand and log
        if ':' not in DEFAULT_CLASSPATH and DEFAULT_CLASSPATH.endswith('/*'):
            classpath_dir = DEFAULT_CLASSPATH[:-2]
            logging.info('java classpath entries: %s' %
                         ', '.join(os.path.join(classpath_dir, filename)
                                   for filename in os.listdir(classpath_dir)))
    except:
        logging.warning('failure logging classpath')
    check_call(args)


def load_accumulo(redis_db, corpus, output_key,
                  batch_size=1000, key_limit=100000, poll_interval=500,
                  java_xmx='10G', irresponsible_gc=False,
                  analytics=DEFAULT_GIGAWORD_ANALYTICS):
    (redis_host, redis_port) = infer_redis_addr(redis_db)

    scion_conf_path = DEFAULT_SCION_CONF_PATH
    log4j2_json_path = DEFAULT_LOG4J2_JSON_PATH
    classpath = DEFAULT_CLASSPATH

    args = scion_java_args(java_xmx, irresponsible_gc, scion_conf_path,
                           log4j2_json_path, classpath) + [
        '%s.RedisLoader' % LOADER_PACKAGE,
        '--corpus', corpus,
        '--key', output_key,
        '--redis-host', redis_host,
        '--redis-port', str(redis_port),
        '--batch-size', str(batch_size),
    ]
    if key_limit is not None:
        args.extend(['--key-limit', str(key_limit)])
    if poll_interval is not None:
        args.extend(['--poll-interval', str(poll_interval)])
    for analytic in analytics:
        args.extend(['--analytic', analytic])

    log_check_call(args)


def load_accumulo_sequential(redis_db, corpus, output_key,
                             batch_size=1000, key_limit=100000,
                             poll_interval=500,
                             java_xmx='10G', irresponsible_gc=False,
                             analytics=DEFAULT_GIGAWORD_ANALYTICS,
                             begin_range=None, end_range=None):
    (redis_host, redis_port) = infer_redis_addr(redis_db)

    scion_conf_path = DEFAULT_SCION_CONF_PATH
    log4j2_json_path = DEFAULT_LOG4J2_JSON_PATH
    classpath = DEFAULT_CLASSPATH

    args = scion_java_args(java_xmx, irresponsible_gc, scion_conf_path,
                           log4j2_json_path, classpath) + [
        '%s.RedisSequentialLoader' % LOADER_PACKAGE,
        '--corpus', corpus,
        '--key', output_key,
        '--redis-host', redis_host,
        '--redis-port', str(redis_port),
        '--batch-size', str(batch_size),
    ]
    for (arg_name, arg_value) in (
            ('--key-limit', key_limit),
            ('--poll-interval', poll_interval),
            ('--begin-range', begin_range),
            ('--end-range', end_range)):
        if arg_value is not None:
            args.extend([arg_name, str(arg_value)])
    for analytic in analytics:
        args.extend(['--analytic', analytic])

    log_check_call(args)


def load_accumulo_lang(redis_db, corpus, output_key,
                       batch_size=1000, key_limit=100000, poll_interval=500,
                       java_xmx='10G', irresponsible_gc=False,
                       analytics=DEFAULT_TWITTER_ANALYTICS, lang='spa'):
    (redis_host, redis_port) = infer_redis_addr(redis_db)

    if corpus != 'twitter':
        raise ValueError(
            'only corpus "twitter" is currently supported for scans by lang')

    scion_conf_path = DEFAULT_SCION_CONF_PATH
    log4j2_json_path = DEFAULT_LOG4J2_JSON_PATH
    classpath = DEFAULT_CLASSPATH

    args = scion_java_args(java_xmx, irresponsible_gc, scion_conf_path,
                           log4j2_json_path, classpath) + [
        '%s.RedisTwitterLanguageIdentificationLoader' % LOADER_PACKAGE,
        '--language', lang,
        '--key', output_key,
        '--redis-host', redis_host,
        '--redis-port', str(redis_port),
        '--batch-size', str(batch_size),
    ]
    if key_limit is not None:
        args.extend(['--key-limit', str(key_limit)])
    if poll_interval is not None:
        args.extend(['--poll-interval', str(poll_interval)])
    for analytic in analytics:
        args.extend(['--analytic', analytic])

    log_check_call(args)


def load_accumulo_lang_batches_by_user(redis_db, corpus, output_key,
                                       batch_size=100, min_batch_size=10,
                                       key_limit=100000,
                                       poll_interval=500, java_xmx='10G',
                                       irresponsible_gc=False,
                                       analytics=DEFAULT_TWITTER_ANALYTICS,
                                       lang='spa'):
    (redis_host, redis_port) = infer_redis_addr(redis_db)

    if corpus != 'twitter':
        raise ValueError(
            'only corpus "twitter" is currently supported for scans by lang')

    scion_conf_path = DEFAULT_SCION_CONF_PATH
    log4j2_json_path = DEFAULT_LOG4J2_JSON_PATH
    classpath = DEFAULT_CLASSPATH

    args = scion_java_args(java_xmx, irresponsible_gc, scion_conf_path,
                           log4j2_json_path, classpath) + [
        '%s.RedisTwitterUserIDIndexLoader' % LOADER_PACKAGE,
        '--minimum-tweets-to-preserve', str(min_batch_size),
        '--language', lang,
        '--key', output_key,
        '--redis-host', redis_host,
        '--redis-port', str(redis_port),
        '--batch-size', str(batch_size),
    ]
    if key_limit is not None:
        args.extend(['--key-limit', str(key_limit)])
    if poll_interval is not None:
        args.extend(['--poll-interval', str(poll_interval)])
    for analytic in analytics:
        args.extend(['--analytic', analytic])

    log_check_call(args)
