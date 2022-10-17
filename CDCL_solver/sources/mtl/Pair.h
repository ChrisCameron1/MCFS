/******************************************************************************************[Heap.h]
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

#ifndef Minisat_Pair_h
#define Minisat_Pair_h

namespace Minisat {

    template <class Type>
    class Pair {
    private:
        Type * first;
        Type * second;
    public:

        Pair() {
            this->first = nullptr;
            this->second = nullptr;
        }

        Pair(Type * first, Type * second) {
            this->first = first;
            this->second = second;
        }

        ~Pair() {
            delete first;
            delete second;
        }

        Type * getFirst() {
            return this->first;
        }

        Type * getSecond() {
            return this->second;
        }

    };
}

#endif