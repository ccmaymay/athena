import os
import codecs
import logging
from urllib import unquote
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from redis import Redis

from concrete.util.file_io import CommunicationReader
from concrete.util.redis_io import RedisCommunicationReader
from concrete.util.simple_comm import create_comm


def mkdirp(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def mkdirp_parent(path):
    dirname = os.path.dirname(path)
    if dirname:
        mkdirp(dirname)


def parse_addr(addr):
    if addr and ':' in addr:
        (host, port_str) = addr.split(':')
        port = int(port_str)
    else:
        raise ValueError('invalid addr: %s' % addr)
    return (host, port)


def parse_redis_addr(url):
    if url:
        if ':' in url:
            (host, port_str) = url.split(':')
            port = int(port_str)
        else:
            host = url
            port = 6379
    else:
        host = 'localhost'
        port = 6379
    return (host, port)


def parse_basic_args(description, cb=None):
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=description,
    )
    parser.add_argument('redis_addr', type=parse_redis_addr,
                        help='host:port of redis database')
    if cb is not None:
        cb(parser)
    ns = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )
    db = Redis(*ns.redis_addr)
    return (db, ns)


def infer_redis_addr(redis_db):
    conn_kw = redis_db.connection_pool.connection_kwargs

    redis_host = conn_kw['host']
    if redis_host is None:
        redis_host = 'localhost'

    redis_port = conn_kw['port']
    if redis_port is None:
        redis_port = 6379

    return (redis_host, redis_port)


class URLCommunicationReader(object):
    def __init__(self, url):
        if url.startswith('file://'):
            path = self.parse_file_url(url)
            self.comms = self.generate_filesystem_comms(path)

        elif url.startswith('textfile://'):
            path = self.parse_file_url(url, 'textfile://')
            self.comms = self.generate_comms_from_filesystem_text(path)

        elif url.startswith('redis://'):
            (host, port, key, params) = self.parse_redis_url(url)
            self.comms = RedisCommunicationReader(Redis(host, port), key,
                                                  **params)

        else:
            path = self.parse_implicit_url(url)
            self.comms = self.generate_filesystem_comms(path)

    @classmethod
    def generate_filesystem_comms(cls, path):
        if os.path.isdir(path):
            for (parent_path, dir_entries, file_entries) in os.walk(path):
                for file_entry in file_entries:
                    file_path = os.path.join(parent_path, file_entry)
                    for (comm, _) in CommunicationReader(file_path):
                        yield comm

        else:
            for (comm, _) in CommunicationReader(path):
                yield comm

    @classmethod
    def generate_comms_from_filesystem_text(cls, path):
        if os.path.isdir(path):
            for (parent_path, dir_entries, file_entries) in os.walk(path):
                for file_entry in file_entries:
                    file_path = os.path.join(parent_path, file_entry)
                    yield cls.filesystem_text_to_comm(file_path)

        else:
            yield cls.filesystem_text_to_comm(path)

    @classmethod
    def filesystem_text_to_comm(cls, path):
        with codecs.open(path, encoding='utf-8') as f:
            return create_comm(path, f.read())

    def __iter__(self):
        for comm in self.comms:
            yield comm

    @classmethod
    def parse_implicit_url(self, url):
        return unquote(url)

    @classmethod
    def parse_file_url(self, url, schema_prefix='file://'):
        path = url[len(schema_prefix):]
        return unquote(path)

    @classmethod
    def parse_redis_url(self, url):
        addr_end = url.index('/', len('redis://'))

        qmark_idx = url.find('?', len('redis://'))
        if qmark_idx < 0:
            key_end = len(url)
            params = dict()
        else:
            key_end = qmark_idx
            params = dict(
                param.split('=')
                for param in url[qmark_idx+1:].split('&')
            )
            for bool_key in ('pop', 'block', 'right_to_left',
                             'cycle_list'):
                if bool_key in params:
                    if params[bool_key].lower().startswith('t'):
                        params[bool_key] = True
                    else:
                        params[bool_key] = False
            for int_key in ('block_timeout', 'temp_key_ttl',
                            'temp_key_leaf_len'):
                if int_key in params:
                    params[int_key] = int(params[int_key])

        addr = url[len('redis://'):addr_end]
        if ':' in addr:
            (host, port_str) = addr.split(':')
            port = int(port_str)
        else:
            host = addr
            port = 6379

        key = unquote(url[addr_end+1:key_end])

        return (host, port, key, params)
