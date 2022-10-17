#ifndef Minisat_LRUCache_h
#define Minisat_LRUCache_h

using namespace Minisat;

class LinkedListNode {
    public:

        LinkedListNode *next;
        LinkedListNode *prev;
        std::vector<bool> key;

        LinkedListNode(std::vector<bool> key, LinkedListNode *next, LinkedListNode *prev) {
            this->key = key;
            this->next = next;
            this->prev = prev;
        }
};

class LRUCache {
    
private:

    std::unordered_map<std::vector<bool>, LinkedListNode *> * dict;
    int capacity;
    int size;
    LinkedListNode *head;
    LinkedListNode *tail;
            
public:

    LRUCache(int capacity) {
        dict = new std::unordered_map<std::vector<bool>, LinkedListNode *>();
        this->capacity = capacity;
        this->size = 0;
        this->head = nullptr;
        this->tail = nullptr;
    }

    std::vector<bool> insert(std::vector<bool> key){
        //printf("Inserting key into cache\n");
        LinkedListNode *node = new LinkedListNode(key, this->head, nullptr);
        // Add node to dictionary
        (*dict)[key] = node;
        // Check if head has not been initialized
        //printf("Checking if head has been initialized\n");
        if (this->head == nullptr) {
            this->head = node;
            this->tail = node;
        } else {
            this->head->prev = node;
            node->next = this->head;
            this->head = node;
        }
        size++;
        if (size > capacity)
        {
            return this->deleteAtBack();
        }
        return (* new std::vector<bool>());
    }

    std::vector<bool> deleteAtBack(){
        LinkedListNode *node = this->tail;
        this->tail = node->prev;
        this->tail->next = nullptr;
        std::vector<bool> deleted_key = node->key;
        this->dict->erase(deleted_key);
        delete node;
        this->size--;
        return deleted_key;
    }

    void moveFromMiddle(std::vector<bool> key){
        //printf("Moving from middle\n");
        LinkedListNode * node = (*this->dict)[key];
        if (node->key != this->head->key)
        {
            //printf("Node key is not head key\n");
            node->prev->next = node->next;
            if (node->next != nullptr)
            {
                node->next->prev = node->prev;
            }
            node->prev = nullptr;
            node->next = this->head;
            this->head->prev = node;
            this->head = node;
        }
    }

    bool exists(std::vector<bool> key){
        //printf("Checking if key exists\n");
        if ((*this->dict).find(key) == (*this->dict).end())
        {
            return false;
        }
        return true;
    }

    int getSize(){
        return this->size;
    }
};

#endif