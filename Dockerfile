FROM centos:7

RUN yum update -y && yum clean all # cache bust 20160712

RUN yum install -y \
        atlas \
        atlas-devel \
        autoconf \
        automake \
        cmake \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        git \
        libtool \
        m4 \
        make \
        numpy \
        pkgconfig \
        python \
        python-devel \
        tar \
        valgrind

WORKDIR /tmp
RUN mkdir -p /usr/local/{include,lib}

RUN curl https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade setuptools && \
    pip install --upgrade \
        cython \
        flake8 \
        pytest \
        pytest-cov \
        gcovr

RUN ln -s /usr/lib64/atlas/libsatlas.so /usr/local/lib/libatlas.so && \
    ln -s /usr/lib64/atlas/libsatlas.so /usr/local/lib/libf77blas.so && \
    ln -s /usr/lib64/atlas/libsatlas.so /usr/local/lib/libcblas.so && \
    ln -s /usr/lib64/atlas/libsatlas.so.3 /usr/local/lib/libatlas.so.3 && \
    ln -s /usr/lib64/atlas/libsatlas.so.3 /usr/local/lib/libf77blas.so.3 && \
    ln -s /usr/lib64/atlas/libsatlas.so.3 /usr/local/lib/libcblas.so.3

RUN git clone https://github.com/google/googletest.git && \
    mkdir gtest-build && \
    pushd gtest-build && \
    cmake ../googletest/googletest && \
    make && \
    mv libgtest.a libgtest_main.a /usr/local/lib/ && \
    mv ../googletest/googletest/include/gtest /usr/local/include/ && \
    popd && \
    mkdir gmock-build && \
    pushd gmock-build && \
    cmake ../googletest/googlemock && \
    make && \
    mv libgmock.a libgmock_main.a /usr/local/lib/ && \
    mv ../googletest/googlemock/include/gmock /usr/local/include/ && \
    popd && \
    rm -rf googletest gtest-build gmock-build

RUN echo '/usr/local/lib' > /etc/ld.so.conf.d/local.conf && \
    ldconfig -v

RUN useradd -m -U -s /bin/bash littleowl && \
    passwd -l littleowl
ADD . /home/littleowl/athena
RUN cd /home/littleowl/athena && \
    python setup.py install && \
    pip install -r test-requirements.txt && \
    python setup.py clean && \
    chown -R littleowl:littleowl /home/littleowl

USER littleowl
WORKDIR /home/littleowl/athena
