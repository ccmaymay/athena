from libcpp.string cimport string
from libcpp.memory cimport shared_ptr


cdef extern from "_serialization.h":
    cdef cppclass FileSerializer[T]:
        Serializer(const string& path)
        void dump(const T& obj) except +
        shared_ptr[T] load() except +
