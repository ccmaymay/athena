#ifndef ATHENA__SERIALIZATION_H
#define ATHENA__SERIALIZATION_H


#include <vector>
#include <unordered_map>
#include <map>
#include <utility>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>


// precision of serialized values
#define SERIALIZATION_PRECISION 9

#define DEFINE_PRIMITIVE_SERIALIZATION(T) \
  template <> \
  struct Serializer<T> { \
    static void serialize(const T& value, std::ostream& stream) { \
      stream << std::setprecision(SERIALIZATION_PRECISION) << \
        value << "\r\n"; \
    } \
    static T deserialize(std::istream& stream) { \
      T value; \
      stream >> value; \
      stream.get(); \
      stream.get(); \
      return T(value); \
    } \
  };


// Stream-based serialization

template <class T>
struct Serializer;


// File-based serialization

template <class T>
class FileSerializer;


//
// Serializer definitions
//


//
// generic
//

template <class T>
struct Serializer {
  static void serialize(const T& value, std::ostream& stream) {
    value.serialize(stream);
  }
  static T deserialize(std::istream& stream) {
    return T::deserialize(stream);
  }
};


//
// primitives
//

DEFINE_PRIMITIVE_SERIALIZATION(short)
DEFINE_PRIMITIVE_SERIALIZATION(int)
DEFINE_PRIMITIVE_SERIALIZATION(long)
DEFINE_PRIMITIVE_SERIALIZATION(float)
DEFINE_PRIMITIVE_SERIALIZATION(double)
DEFINE_PRIMITIVE_SERIALIZATION(size_t)
DEFINE_PRIMITIVE_SERIALIZATION(char)
DEFINE_PRIMITIVE_SERIALIZATION(bool)


//
// pair
//

template <class K, class V>
struct Serializer<std::pair<K,V> > {
  static void serialize(const std::pair<K,V>& container, std::ostream& stream) {
    Serializer<K>::serialize(container.first, stream);
    Serializer<V>::serialize(container.second, stream);
  }
  static std::pair<K,V> deserialize(std::istream& stream) {
    auto first(Serializer<K>::deserialize(stream));
    auto second(Serializer<V>::deserialize(stream));
    return std::pair<K,V>(
      std::move(first),
      std::move(second)
    );
  }
};


//
// string
//

template <>
struct Serializer<std::string> {
  static void serialize(const std::string& container, std::ostream& stream) {
    Serializer<size_t>::serialize(container.size(), stream);
    for (size_t i = 0; i < container.size(); ++i) {
      stream << container[i];
    }
  }
  static std::string deserialize(std::istream& stream) {
    auto size(Serializer<size_t>::deserialize(stream));
    std::string container(size, 0);
    for (size_t i = 0; i < size; ++i) {
      stream >> container[i];
    }
    return container;
  }
};


//
// unordered_map
//

template <class K, class V>
struct Serializer<std::unordered_map<K,V> > {
  static void serialize(const std::unordered_map<K,V>& container, std::ostream& stream) {
    Serializer<size_t>::serialize(container.size(), stream);
    for (auto it = container.cbegin();
         it != container.cend();
         ++it) {
      Serializer<std::pair<K,V> >::serialize(*it, stream);
    }
  }
  static std::unordered_map<K,V> deserialize(std::istream& stream) {
    auto size(Serializer<size_t>::deserialize(stream));
    std::unordered_map<K,V> container;
    for (size_t i = 0; i < size; ++i) {
      container.insert(
        Serializer<std::pair<K,V> >::deserialize(stream));
    }
    return container;
  }
};


//
// multimap
//

template <class K, class V>
struct Serializer<std::multimap<K,V> > {
  static void serialize(const std::multimap<K,V>& container, std::ostream& stream) {
    Serializer<size_t>::serialize(container.size(), stream);
    for (auto it = container.cbegin();
         it != container.cend();
         ++it) {
      Serializer<std::pair<K,V> >::serialize(*it, stream);
    }
  }
  static std::multimap<K,V> deserialize(std::istream& stream) {
    auto size(Serializer<size_t>::deserialize(stream));
    std::multimap<K,V> container;
    for (size_t i = 0; i < size; ++i) {
      container.insert(
        Serializer<std::pair<K,V> >::deserialize(stream));
    }
    return container;
  }
};


//
// vector
//

template <class T>
struct Serializer<std::vector<T> > {
  static void serialize(const std::vector<T>& container, std::ostream& stream) {
    Serializer<size_t>::serialize(container.size(), stream);
    for (auto it = container.cbegin();
         it != container.cend();
         ++it) {
      Serializer<T>::serialize(*it, stream);
    }
  }
  static std::vector<T> deserialize(std::istream& stream) {
    auto size(Serializer<size_t>::deserialize(stream));
    std::vector<T> container;
    container.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      container.push_back(Serializer<T>::deserialize(stream));
    }
    return container;
  }
};


//
// FileSerializer definitions
//


template <class T>
class FileSerializer final {
  std::string _path;

  public:
    FileSerializer(const std::string& path): _path(path) { }

    void dump(const T& obj) const {
      std::ofstream output_file;
      output_file.open(_path.c_str());
      if (output_file) {
        Serializer<T>::serialize(obj, output_file);
        output_file.close();
      } else {
        throw std::runtime_error(
          std::string("output file ") + _path +
          std::string(" cannot be written"));
      }
    }

    T load() const {
      std::ifstream input_file;
      input_file.open(_path.c_str());
      if (input_file) {
        auto obj(Serializer<T>::deserialize(input_file));
        input_file.close();
        return obj;
      } else {
        throw std::runtime_error(
          std::string("input file ") + _path +
          std::string(" cannot be read"));
      }
    }
};


#endif
