from athena.util import URLCommunicationReader


def test_parse_file_url():
    assert 'foo/bar' == URLCommunicationReader.parse_file_url(
        'file://foo/bar')
    assert 'foo /%bar' == URLCommunicationReader.parse_file_url(
        'file://foo%20/%25bar')
    assert '/foo/bar' == URLCommunicationReader.parse_file_url(
        'file:///foo/bar')
    assert '/foo /%bar' == URLCommunicationReader.parse_file_url(
        'file:///foo%20/%25bar')


def test_parse_implicit_url():
    assert 'foo/bar' == URLCommunicationReader.parse_implicit_url(
        'foo/bar')
    assert 'foo /%bar' == URLCommunicationReader.parse_implicit_url(
        'foo%20/%25bar')
    assert '/foo/bar' == URLCommunicationReader.parse_implicit_url(
        '/foo/bar')
    assert '/foo /%bar' == URLCommunicationReader.parse_implicit_url(
        '/foo%20/%25bar')


def test_parse_redis_url():
    assert (
        'test5', 123, 'foo/bar', dict()
    ) == URLCommunicationReader.parse_redis_url(
        'redis://test5:123/foo/bar'
    )
    assert (
        'test5', 6379, 'foo/bar', dict()
    ) == URLCommunicationReader.parse_redis_url(
        'redis://test5/foo/bar'
    )
    assert (
        'test5', 123, 'foo /%bar', dict()
    ) == URLCommunicationReader.parse_redis_url(
        'redis://test5:123/foo%20/%25bar'
    )
    assert (
        'test5', 123, 'foo/bar', dict(pop=False, block=True, block_timeout=7)
    ) == URLCommunicationReader.parse_redis_url(
        'redis://test5:123/foo/bar?pop=False&block=True&block_timeout=7',
    )
    assert (
        'test5', 123, '/foo/bar', dict()
    ) == URLCommunicationReader.parse_redis_url(
        'redis://test5:123//foo/bar'
    )
