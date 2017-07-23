# athena

[![Build Status](https://travis-ci.org/cjmay/athena.svg?branch=master)](https://travis-ci.org/cjmay/athena)

Athena is a library and collection of programs implementing streaming
embeddings with the space-saving algorithm.  Code released for the
["Streaming Word Embeddings with the Space-Saving Algorithm"
(arXiv:1704.07463) manuscript](https://arxiv.org/abs/1704.07463)
can be found in the
[arxiv-1704-07463v1 tag](https://github.com/cjmay/athena/tree/arxiv-1704-07463v1).

Athena comprises a C++ library and a few programs.
The source code is located in `src/`.

## Usage

To build the library by itself run `make lib`.
To build the programs run `make main`.  The
compiled library and programs will be placed in `build/lib/`.

## Testing

The test suite requires Google Test and Google Mock.  On Linux or
OS X, install cmake (e.g., with `sudo yum install cmake` or
`sudo apt-get install cmake`) and then run the following bash shell
code to install Google Test and Google Mock:

```bash
git clone https://github.com/google/googletest.git && \
    mkdir gtest-build && \
    pushd gtest-build && \
    cmake ../googletest/googletest && \
    make && \
    sudo mv libgtest.a libgtest_main.a /usr/local/lib/ && \
    sudo mv ../googletest/googletest/include/gtest \
        /usr/local/include/ && \
    popd && \
    mkdir gmock-build && \
    pushd gmock-build && \
    cmake ../googletest/googlemock && \
    make && \
    sudo mv libgmock.a libgmock_main.a /usr/local/lib/ && \
    sudo mv ../googletest/googlemock/include/gmock \
        /usr/local/include/ && \
    popd && \
    rm -rf googletest gtest-build gmock-build
```

Now to run the tests do: `make test`

## References

1.  [Walker (1977)](http://dl.acm.org/citation.cfm?doid=355744.355749)
2.  [Wikipedia alias method article](https://en.wikipedia.org/wiki/Alias_method)
3.  [Vitter (1985)](https://www.cs.umd.edu/~samir/498/vitter.pdf)
4.  [Metwally, Agrawal, and El Abaddi (2005)](http://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf)
5.  [Cormode (2009) slides](http://dmac.rutgers.edu/Workshops/WGUnifyingTheory/Slides/cormode.pdf)
6.  [Knoll space saving implementation article](http://byronknoll.blogspot.com/2013/01/space-saving-algorithm.html)
7.  [D'Elia (2013) (Section 3)](http://www.dis.uniroma1.it/~delia/files/docs/spacesaving-report.pdf)
8.  [Mikolov, Sutskever, Chen, Corrado, and Dean (2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
9.  [Levy, Goldberg, and Dagan (2015)](http://www.aclweb.org/anthology/Q15-1016)
10. [Mikolov, Sutskever, Chen, Corrado, and Dean (2013) word2vec implementation](https://code.google.com/archive/p/word2vec/)
11. [Řehůřek gensim word2vec implementation article (part 1)](http://rare-technologies.com/deep-learning-with-word2vec-and-gensim/)
12. [Řehůřek gensim word2vec implementation article (part 2)](http://rare-technologies.com/word2vec-in-python-part-two-optimizing/)
13. [Řehůřek gensim word2vec implementation article (part 3)](http://rare-technologies.com/parallelizing-word2vec-in-python/)

## License

Copyright 2012-2017 Johns Hopkins University HLTCOE. All rights
reserved.  This software is released under the 2-clause BSD license.
See LICENSE in this directory for more information.
