# athena

Athena is a library and collection of scripts implementing streaming
embeddings with the space-saving algorithm.  See the accompanying
manuscript at [arXiv:1704.07463](https://arxiv.org/abs/1704.07463).

Athena comprises a C++ library accessible in Python using Cython.
The source code is located in `athena/` with some general-purpose
scripts located in `scripts/`.  More specialized scripts implementing
the Twitter experiments from the paper are provided in the
`streaming-embeddings/` subdirectory.  Finally, source code for a few
simple C++ programs is included under `athena/` along with the library
code.

## Usage

The athena Python interface must be installed prior to usage.  To
install it, first install the build dependencies Cython and numpy, if
you have not done so already, with `pip install cython numpy`.  Then do
`python setup.py install` to build and install athena and runtime
dependencies.

To build the simple, standalone C++ programs run `make main`.  The
compiled programs will be placed in `build/lib/`.

## Testing

As with the athena library and scripts, the athena tests comprise a C++
component and a Python component.  Most tests are written in C++.

The C++ test suite requires Google Test and Google Mock.  On Linux or
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

To run the C++ tests do: `make test`

To run the Python tests, install athena, install test requirements with
`pip install -r test-requirements.txt`, then do: `py.test`

## References

1.  [Mitzenmacher, Steinke, and Thaler (2001) (Section 2)](http://arxiv.org/abs/1102.5540)
2.  [Řehůřek gensim word2vec implementation article (part 1)](http://rare-technologies.com/deep-learning-with-word2vec-and-gensim/)
3.  [Řehůřek gensim word2vec implementation article (part 2)](http://rare-technologies.com/word2vec-in-python-part-two-optimizing/)
4.  [Řehůřek gensim word2vec implementation article (part 3)](http://rare-technologies.com/parallelizing-word2vec-in-python/)
5.  [Řehůřek gensim word2vec tutorial](http://rare-technologies.com/word2vec-tutorial/)
6.  [Knoll space saving implementation article](http://byronknoll.blogspot.com/2013/01/space-saving-algorithm.html)
7.  [Crammer, Dekel, Keshet, Shalev-Shwartz, and Singer (2006)](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
8.  [Mikolov, Sutskever, Chen, Corrado, and Dean (2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
9.  [D'Elia (2013) (Section 3)](http://www.dis.uniroma1.it/~delia/files/docs/spacesaving-report.pdf)
10. [Mikolov, Sutskever, Chen, Corrado, and Dean (2013) word2vec implementation](https://code.google.com/archive/p/word2vec/)
11. [Metwally, Agrawal, and El Abaddi (2005)](http://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf)
12. [Cormode (2009) slides](http://dmac.rutgers.edu/Workshops/WGUnifyingTheory/Slides/cormode.pdf)
13. [Levy, Goldberg, and Dagan (2015)](http://www.aclweb.org/anthology/Q15-1016)
14. [Rastogi, Costello, Levin LSH implementation](https://gitlab.hltcoe.jhu.edu/prastog3/lsh)

## License

Copyright 2012-2017 Johns Hopkins University HLTCOE. All rights
reserved.  This software is released under the 2-clause BSD license.
See LICENSE in this directory for more information.
