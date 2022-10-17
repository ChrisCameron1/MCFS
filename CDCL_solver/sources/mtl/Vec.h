/*******************************************************************************************[Vec.h]
Copyright (c) 2003-2007, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson
Copyright (c) 2018, Chris Cameron

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 **************************************************************************************************/

#ifndef Minisat_Vec_h
#define Minisat_Vec_h

#include <assert.h>
#include <new>

#include "mtl/IntTypes.h"
#include "mtl/XAlloc.h"

#include "stdio.h"

namespace Minisat {

    //=================================================================================================
    // Automatically resizable arrays
    //
    // NOTE! Don't use this vector on datatypes that cannot be re-located in memory (with realloc)

    template<class T>
    class vec {
        T* data;
        int sz;
        int cap;

        // Don't allow copying (error prone):
        // vec<T>&  operator = (const vec<T>& other) { //assert(0); return *this;
        //                                             other.copyTo(*this);
        //                                             // for (int i = 0; i < other.size(); i++)
        //                                             // {
        //                                             //     this->push(other[i]);
        //                                             // } 
        //                                     }

        vec(vec<T>& other) {
            assert(0);
        }

        // Helpers for calculating next capacity:

        static inline int imax(int x, int y) {
            int mask = (y - x) >> (sizeof (int)*8 - 1);
            return (x & mask) + (y & (~mask));
        }
        //static inline void nextCap(int& cap){ cap += ((cap >> 1) + 2) & ~1; }

        static inline void nextCap(int& cap) {
            cap += ((cap >> 1) + 2) & ~1;
        }

    public:
        // Constructors:

        vec() : data(NULL), sz(0), cap(0) {
        }

        explicit vec(int size) : data(NULL), sz(0), cap(0) {
            growTo(size);
        }

        vec(int size, const T& pad) : data(NULL), sz(0), cap(0) {
            growTo(size, pad);
        }

        ~vec() {
            clear(true);
        }

        // Pointer to first element:

        operator T* (void) {
            return data;
        }

        // Size operations:

        int size(void) const {
            return sz;
        }

        void shrink(int nelems) {
            assert(nelems <= sz);
            for (int i = 0; i < nelems; i++) sz--, data[sz].~T();
        }

        void shrink_(int nelems) {
            assert(nelems <= sz);
            sz -= nelems;
        }

        int capacity(void) const {
            return cap;
        }
        void capacity(int min_cap);
        void growTo(int size);
        void growTo(int size, const T& pad);
        void clear(bool dealloc = false);

        // Stack interface:

        void push(void) {
            if (sz == cap) capacity(sz + 1);
            new (&data[sz]) T();
            sz++;
        }

        void push(const T& elem) {
            assert(sz <= cap);
            if (sz == cap) {
                capacity(sz + 1);
            }
            data[sz++] = elem;
        }

        /*
            Append two vec<>
         */
        void push(const vec<T>& elems) {
            for (int i = 0; i < elems.size(); i++) {
                push(elems[i]);
            }
        }

        void push_(const T& elem) {
            assert(sz < cap);
            data[sz++] = elem;
        }

        void pop(void) {
            assert(sz > 0);
            sz--, data[sz].~T();
        }
        // NOTE: it seems possible that overflow can happen in the 'sz+1' expression of 'push()', but
        // in fact it can not since it requires that 'cap' is equal to INT_MAX. This in turn can not
        // happen given the way capacities are calculated (below). Essentially, all capacities are
        // even, but INT_MAX is odd.

        const T& last(void) const {
            return data[sz - 1];
        }

        T& last(void) {
            return data[sz - 1];
        }

        // Vector interface:

        const T& operator[](int index) const {
            return data[index];
        }

        T& operator[](int index) {
            return data[index];
        }

        /** 
         *   @brief  Override assigns operator to assign vec to another by value. Note that this can be dangerous to overload like this
         *   @param  other is vec<> to assigned to this object 
         * 
         *   @return updated vec<>
         */
        vec<T>& operator=(const vec<T>& other) {
            this->sz = 0;
            this->cap = 0;
            this->data = nullptr;

            for (int i = 0; i < other.size(); i++) {

                this->push(other[i]);
            }
            // Make sure the copied object has the same size
            assert(other.size() == this->size());
            return *this;
        }

        // Duplication (preferred instead):

        void copyTo(vec<T>& copy) const { //printf("CopyTo vec<> function\n");
            fflush(stdout);
            copy.clear();
            copy.growTo(sz);
            //printf("CopyTo vec<> finished growTO\n");
            fflush(stdout);
            for (int i = 0; i < sz; i++) copy[i] = data[i];
        }

        void moveTo(vec<T>& dest) {
            dest.clear(true);
            dest.data = data;
            dest.sz = sz;
            dest.cap = cap;
            data = NULL;
            sz = 0;
            cap = 0;
        }
    };

    template<class T>
    void vec<T>::capacity(int min_cap) {
        if (cap >= min_cap) return;
        int add = imax((min_cap - cap + 1) & ~1, ((cap >> 1) + 2) & ~1); // NOTE: grow by approximately 3/2
        // What if sizeof(T) is not fixed. We have a vec<> for T which could be of arbitrary size,
        // I could do sizeof(expression), where expression is constructor type T of necessary size
        //printf("Mem to add: %d, cap: %d. Size to be allocated: %d\n",add, cap, (add+cap)*sizeof(T));
        // Below: pointer being realloc'd was not allocated
        // Data is pointer: so it should get properly realloc'd
        if (add > INT_MAX - cap || ((data = (T*)::realloc(data, (cap += add) * sizeof (T))) == NULL) && errno == ENOMEM) {
            //printf("Mem to add: %d, cap: %d\n",add, cap);
            fflush(stdout);
            throw OutOfMemoryException();
        }
        //printf("Successfully reallocated to capacity\n");
    }

    template<class T>
    void vec<T>::growTo(int size, const T& pad) {
        //printf("GrowTo function");
        if (sz >= size) return;
        capacity(size);
        for (int i = sz; i < size; i++) {
            data[i] = pad;
            //printf("Pad added: %d, data[i]: %d\n", pad, data[i]);
        }
        sz = size;
    }

    template<class T>
    void vec<T>::growTo(int size) {
        //printf("GrowTo function");
        if (sz >= size) return;
        capacity(size);
        for (int i = sz; i < size; i++) new (&data[i]) T();
        sz = size;
    }

    template<class T>
    void vec<T>::clear(bool dealloc) {
        if (data != NULL) {
            for (int i = 0; i < sz; i++) data[i].~T();
            sz = 0;
            if (dealloc) free(data), data = NULL, cap = 0;
        }
    }

    //=================================================================================================
}

#endif
