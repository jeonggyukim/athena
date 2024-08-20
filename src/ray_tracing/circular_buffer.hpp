#ifndef RAY_TRACING_CIRCULAR_BUFFER_HPP_
#define RAY_TRACING_CIRCULAR_BUFFER_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file circular_buffer.hpp
//! \brief Provides CircularBuffer class template for storing objects
//!
//! Copy constructor and assignment operator are disabled.
//!=======================================================================================

// C++ headers
#include <cstring>    // memcpy
#include <stdexcept>  // runtime_error

using byte = unsigned char;

//--------------------------------------------------------------------------------------
//! \class CircularBuffer
//! \brief Defines the class for storing objects in a circular buffer

template <typename T>
class CircularBuffer {
 public:
  // Unlike AthenaArray, always allocate memory for array_
  explicit CircularBuffer(int maxsize=1) :
    maxsize_(maxsize), front_(-1), rear_(-1), size_(0) { array_ = new T[maxsize_]; }
  // Delete copy constructor and assignment operator
  CircularBuffer(const CircularBuffer&) = delete;
  CircularBuffer& operator=(const CircularBuffer&) = delete;
  ~CircularBuffer();

  void Resize(); //!> Increase array size
  void Push(T *x); //!> Add one element to rear
  void PushMultiple(byte *buf, int n); //!> Add multiple elements
  void Reset();
  T* PopFront();
  T* PopRear();

  int GetSize() const { return size_; }
  int GetMaxSize() const { return maxsize_; }
  T* GetArray() const { return array_; }

 private:
  int maxsize_;
  int front_;
  int rear_;
  int size_;
  T* array_;
};

//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::CircularBuffer()
//! \brief CircularBuffer contructor

// template<typename T>
// CircularBuffer<T>::CircularBuffer(int maxsize) :
//   maxsize_(maxsize), front_(-1), rear_(-1), size_(0) {
//   array_ = new T[maxsize_];
// }

//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::~CircularBuffer()
//! \brief CircularBuffer destructor

template<typename T>
CircularBuffer<T>::~CircularBuffer() {
  delete[] array_;
}

//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::Resize()
//! \brief Increase array size

template<typename T>
void CircularBuffer<T>::Resize() {
  if (front_ > rear_) { // Case 1: head isn't ahead of tail
    // Relocate array[0:rear] to array[maxsize:maxsize+rear]
    T *buf = new T[rear_ + 1];
    std::memcpy(buf, array_, (rear_+1)*sizeof(T));
    T *new_array = new T[2*maxsize_];
    if (new_array == nullptr) {
      throw std::runtime_error("[CircularBuffer::Resize]: "
                               "realloc returned a NULL pointer, "
                               "new maxsize: " + std::to_string(maxsize_));
    }
    std::memcpy(new_array + maxsize_, buf, (rear_+1)*sizeof(T));
    delete[] array_;
    array_ = new_array;
    rear_ += maxsize_;
    delete[] buf;
  } else { // Case 2: Head is ahead of tail
    T *new_array = new T[2*maxsize_];
    if (new_array == nullptr) {
      throw std::runtime_error("[CircularBuffer::Resize]: "
                               "realloc returned a NULL pointer, "
                               "new maxsize: " + std::to_string(2*maxsize_));
    }
    std::memcpy(new_array, array_, maxsize_*sizeof(T));
    delete[] array_;
    array_ = new_array;
  }
  maxsize_ *= 2;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::Push()
//! \brief Add one element to tail

template<typename T>
void CircularBuffer<T>::Push(T *x) {
  if (size_ == maxsize_) {  // array at full capacity
    Resize();
    Push(x);
  } else {
    rear_++;
    rear_ = (rear_ == maxsize_) ? 0 : rear_;
    array_[rear_] = *x;
    size_++;
    if (front_ == -1) {
      front_ = rear_;
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::PushMultiple()
//! \brief Add multiple elements.

template<typename T>
void CircularBuffer<T>::PushMultiple(byte *buf, int n) {
  while (maxsize_ < size_ + n) Resize();
  if (rear_ >= front_) {
    // n1 : length from rear to maxsize-1
    int n1 = maxsize_ - rear_ - 1;
    if (n1 >= n) { // copy all at once
      std::memcpy(reinterpret_cast<byte *>(&(array_[rear_+1])),
                  buf, n*sizeof(T));
      rear_ += n;
    } else {       // copy separately
      int n2 = n - n1;
      std::memcpy(reinterpret_cast<byte *>(&(array_[rear_+1])),
                  buf, n1*sizeof(T));
      std::memcpy(reinterpret_cast<byte *>(array_),
                  &buf[n1*sizeof(T)], n2*sizeof(T));
      rear_ = n2 - 1;
    }
  } else {
    std::memcpy(reinterpret_cast<byte *>(&(array_[rear_+1])),
                buf, n*sizeof(T));
    rear_ += n;
  }
  if (front_ == -1) front_ = 0;
  size_ += n;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::Reset()
//! \brief Set the container size to zero. Maxsize and array remains the same.

template<typename T>
void CircularBuffer<T>::Reset() {
  size_ = 0;
  front_ = -1;
  rear_ = -1;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::PopFront()
//! \brief Pops one element at front.

template<typename T>
T* CircularBuffer<T>::PopFront() {
  T *x = nullptr;
  if (size_ == 0) {
    throw std::runtime_error("[CircularBuffer::PopFront] size is zero!\n");
  } else {
    x = &(array_[front_]);
    if(size_ == 1) {
      front_ = -1;
      rear_ = -1;
    } else {
      front_++;
      front_ = (front_ == maxsize_) ? 0 : front_;
    }
    size_--;
  }
  return x;
}

//--------------------------------------------------------------------------------------
//! \fn CircularBuffer::PopRear()
//! \brief Pops one element at rear.

template<typename T>
T* CircularBuffer<T>::PopRear() {
  T *x = nullptr;
  if (size_ == 0) {
    throw std::runtime_error("[CircularBuffer::PopRear] size is zero!\n");
  } else {
    x = &(array_[rear_]);
    if (size_ == 1) {
      front_ = -1;
      rear_ = -1;
    } else {
      rear_--;
      rear_ = (rear_ == -1) ? maxsize_ : rear_;
    }
    size_--;
  }
  return x;
}
#endif  // RAY_TRACING_CIRCULAR_BUFFER_HPP_
