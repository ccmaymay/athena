from libcpp.vector cimport vector


cdef extern from "_math.h":
    cdef cppclass CountNormalizer:
        CountNormalizer(float exponent, float offset)

    cdef cppclass ReservoirSampler[T]:
        ReservoirSampler(size_t reservoir_size)
        T sample()
        T insert(T val)
        const T& operator[](size_t idx)
        size_t size()
        size_t filled_size()
