FROM centos:7

RUN yum update -y && yum clean all # cache bust 20160712

RUN yum install -y \
        autoconf \
        automake \
        cmake \
        curl \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        git \
        libtool \
        m4 \
        make \
        pkgconfig \
        python \
        python-devel \
        tar \
        valgrind

WORKDIR /tmp
RUN mkdir -p /usr/local/{include,lib}

RUN curl https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade setuptools && \
    pip install --upgrade gcovr

RUN curl -L http://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz | tar -xz && \
    cd OpenBLAS-0.2.19 && \
    make && \
    make install PREFIX=/usr/local && \
    cd .. && \
    rm -rf OpenBLAS-0.2.19

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
RUN chown -R littleowl:littleowl /home/littleowl

USER littleowl
WORKDIR /home/littleowl/athena
