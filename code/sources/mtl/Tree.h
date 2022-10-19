/******************************************************************************************[Tree.h]
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

#ifndef Minisat_Tree_h
#define Minisat_Tree_h

#include "mtl/Node.h"
#include "mtl/Vec.h"

#include <stdio.h>

namespace Minisat {

    template <class Type>
    class Tree : public Node<Type> {
    private:
        vec<Tree<Type> * > * children;
        Tree<Type> * parent;
        torch::Tensor neural_net_lit_probabilities;
    public:

        Tree() : Node<Type>() {
            this->children = nullptr;
            this->parent = nullptr;
            this->neural_net_lit_probabilities = at::zeros(1);
        }

        Tree(int name, Type * value) : Node<Type>(name, value) {
            this->children = new vec<Tree<Type> * >();
            this->parent = nullptr;
            this->neural_net_lit_probabilities =  at::zeros(1);
        }

        ~Tree(){

            for (int i = 0; i<this->children->size(); i++) {
                Tree<Type> * child = (*(this->children))[i];
                delete child;
                // Delete node
            }

        }

        void setChild(Tree<Type> * child) {
            this->children->push(child);
        }

        void setParent(Tree<Type> * parent) {
            this->parent = parent;
        }

        vec<Tree<Type> *> * getChildren() {
            return this->children;
        }

        /** 
         *   @brief  Get child node from this tree which matches the name
         *   @param  name is name of child to be returned  
         * 
         *   @return child node from this tree which matches the name
         */
        Tree<Type> * getChild(int name) {
            if (!this->children)
                return nullptr;
            if (this->children->size() == 0)
                return nullptr;

            for (int i = 0; i<this->children->size(); i++) {
                Tree<Type> * child = (*(this->children))[i];
                if (name == child->getName()) {
                    return child;
                }
            }
            // Child does not exist
            //printf("Child does not exist for literal %d!\n", name);
            return nullptr;
        }

        Tree<Type> * getParent() {
            return this->parent;
        }

        /** 
         *   @brief  Get all nodes in tree  
         * 
         *   @return all nodes in tree
         */
        vec<Tree<Type> *> * get_all_points() {
            vec<Tree<Type> * > * data_points = new vec<Tree<Type> * >();
            data_points->push(this);
            for (int i = 0; i < this->children->size(); i++) {
                vec<Tree<Type> * > * child_points = (*(this->children))[i]->get_all_points();
                data_points->push(*child_points);
            }
            return data_points;
        }

        torch::Tensor get_neural_net_lit_probabilities()
        {
            return this->neural_net_lit_probabilities;
        }

        void set_neural_net_lit_probabilities(torch::Tensor a_neural_net_lit_probabilities)
        {
            this->neural_net_lit_probabilities = a_neural_net_lit_probabilities;
        }

    };
}

#endif
