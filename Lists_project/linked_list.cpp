// Program for creating **singly** linked list

#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

struct Node {
    /* Struct for nodes in linked list. Carries value (int) and pointer to 
    next node in list.*/

    int value;
    Node* next;

    Node(int val) {
        value = val;
        next = nullptr;
    }

    Node(int val, Node* nex) {
        value = val;
        next = nex;
    }
};

class LinkedList {
    /* List class using linked Nodes for storing data 

    List can take new values at end (append), values can be inserted, removed,
    popped and popped from end without index. List can be indexed, length of
    list can be retrieved, and the list can be pretty printed. List uses 
    dynamic memory allocation, and has destructor.*/

private:
    Node* head;
    Node* tail;
    int size;

public:
    LinkedList() {
        head = nullptr;
        tail = nullptr;
        size = 0;
    }

    LinkedList(vector<int> invec) {
        size = invec.size();
        head = new Node(invec[0]);
        Node* current = head;
        for (int i=0;i<size-1;i++) {
            current -> next = new Node(invec[i+1]);
            current = current -> next;
        }
        tail = current;
    }

    ~LinkedList() {
        if (size == 0) {
            ;
        } else if (size == 1) {
            delete head;
        } else if (size == 2) {
            delete head;
            delete tail;
        } else {
            Node* current = head;
            Node* next = current -> next;

        while (current != nullptr) {
            next = current -> next;
            delete current;
            current = next;
            
            }
        }
    }

    int length() {
        return size;
    }

    void append(int val) {
        if (size == 0) {
            head = new Node(val);
            tail = head;
        } else {
            tail -> next = new Node(val);
            tail = tail -> next;
            }
        size++;
    }

    int& get(int index) {
        if (index >= size or index < 0) {
            throw out_of_range("IndexError");
        } else if (index == 0) {
            return head -> value;
        } else if (index == size - 1) {
            return tail -> value;
        } else {
            Node* current = head;

            for (int i=0; i<index; i++) {
                current = current -> next;
            }
            return current -> value;
            }
    }

    int& operator[](int index) {
        return get(index);
    }

    void insert(int val, int index) {
        if (index > size or index < 0) {
            throw out_of_range("IndexError");

        } else if (index == 0) {
            Node* current = head;
            head = new Node(val,current);

        } else if (index == size) {
            append(val);

        } else {        
            Node* current = head;
            for (int i=0;i<index-1;i++) {
                current = current -> next;
            }
            Node* splice = current ->next;
            current -> next = new Node(val,splice);
        }
        size++;
    }

    void remove(int index) {        
        if (index >= size or index < 0) {
            throw out_of_range("IndexError");

        } else if (index == 0) {
            if (size == 1) {
                delete head;

            } else {
                Node* current = head;
                head = current -> next;
                delete current;

            }

        } else if (index == 1) {
            Node* current = head -> next;
            head -> next = current -> next;
            delete current;

            if (size == 2) {
                tail = head;
            }

        } else {
            Node* current = head;

            for (int i=0;i<index-1;i++) {
                current = current -> next;
            }

            Node* trash = current -> next;
            Node* splice = trash -> next;

            current -> next = splice;
            delete trash;
            if (index == size - 1) {
                tail = current;
            }
            }
        size--;
    }

    int pop(int index) {
        int ret = get(index);
        remove(index);
        return ret;
    }

    int pop() {
        return pop(size-1);
    }

    void print() {
        if (size == 0) {
            cout << "[empty]" << endl;
        } else {
            Node* current = head;

            cout << "[";

            while (current -> next != nullptr) {
                cout << current -> value;
                cout << ", ";
                current = current -> next;
            }

            cout << current->value << "]" << endl;
        }
    }

};

void test_div() {
    /* A test function for evaluating functionality of LinkedList class*/

    cout << "\nEntering test mode:\n" << endl;
    cout << "Testing class LinkedList for methods and functionality\n" << endl;

    vector<int> invec{1,8,3,9,3,2,7,9,0,6};
    LinkedList test(invec);

    cout << "Initializing LinkedList from vector:" << endl;
    test.print();

    cout << "\nPopping elements from end of list:" << endl;
    test.print();
    for (int i=0;i<10;i++) {
        cout << test.pop() << endl;
        test.print();
    }

    cout << "\nAppending numbers 0 to 5 to list:" << endl;
    for (int i=0;i<6;i++) {
        test.append(i);
        test.print();
    }

    cout << "\nPopping numbers from place number 0:" << endl;
    test.print();

    for (int i=0;i<6;i++) {
        cout << test.pop(0) << endl;
        test.print();
    }

    for (int i=0;i<6;i++) {
        test.append(i);
    }

    cout << "\nNew list initiated by append:" << endl;
    test.print();

    cout << "\nPopping numbers from place number 1:" << endl;
    test.print();
    for (int i=0;i<6-1;i++) {
        cout << test.pop(1) << endl;
        test.print();
   }
    cout << "\nPopping last item from place number 0:" << endl;
    test.print();
    cout << test.pop(0) << endl;;
    test.print();

    for (int i=6;i>0;i--) {
        test.append(i);
    }

    cout << "\nNew list initiated by append:" << endl;
    test.print();

    cout << "\nPopping numbers from middle of list:" << endl;
    test.print();

    for (int i=6;i>3;i--) {
        cout << test.pop(3) << endl;
        test.print();
    }

    cout << "\nPopping last items from place number 2, 1 and 0:" << endl;
    test.print();
    cout << test.pop(2) << endl;
    test.print();
    cout << test.pop(1) << endl;
    test.print();
    cout << test.pop(0) << endl;;
    test.print();

    for (int i=0;i<6;i++) {
        test.append(i);
    }

    cout << "\nNew list initiated by append:" << endl;
    test.print();

    cout << "\nRemoving numbers from end w/o popping:" << endl;
    test.print();

    for (int i=0;i<6;i++) {
        test.remove(5-i);
        test.print();
    }

    LinkedList test2;

    cout << "\nNew list initiated as empty:" << endl;
    test2.print();

    for (int i=8;i>0;i--) {
        test2.append(i);
    }

    cout << "\nAppended 8 numbers to clean list:" << endl;
    test2.print();

    cout << "\nInserting same numbers at every second step:" << endl;
    test2.print();
    for (int i=0;i<8;i++) {
        test2.insert(8-i,2*i);
        test2.print();
    }

    test2[3] = 100;

    cout << "\nHave changed item number 3 to 100 by indexing."; 
    cout << "Trying to retrieve same item by indexing: \n";
    cout << "Item 3 is " << test2[3] << endl;

    cout << "\nRetrieving length from list by function as: ";
    cout << test2.length();
    cout << " items (16 is correct)" << endl;

    cout << "\nWe now conclude testing after having tested all methods\n"; 
    cout << endl;
}

int main () {
    /* Initialized tests as a separate test function for structure*/

    test_div();

     return 0;
}

/*
Output:

Entering test mode:

Testing class LinkedList for methods and functionality

Initializing LinkedList from vector:
[1, 8, 3, 9, 3, 2, 7, 9, 0, 6]

Popping elements from end of list:
[1, 8, 3, 9, 3, 2, 7, 9, 0, 6]
6
[1, 8, 3, 9, 3, 2, 7, 9, 0]
0
[1, 8, 3, 9, 3, 2, 7, 9]
9
[1, 8, 3, 9, 3, 2, 7]
7
[1, 8, 3, 9, 3, 2]
2
[1, 8, 3, 9, 3]
3
[1, 8, 3, 9]
9
[1, 8, 3]
3
[1, 8]
8
[1]
1
[empty]

Appending numbers 0 to 5 to list:
[0]
[0, 1]
[0, 1, 2]
[0, 1, 2, 3]
[0, 1, 2, 3, 4]
[0, 1, 2, 3, 4, 5]

Popping numbers from place number 0:
[0, 1, 2, 3, 4, 5]
0
[1, 2, 3, 4, 5]
1
[2, 3, 4, 5]
2
[3, 4, 5]
3
[4, 5]
4
[5]
5
[empty]

New list initiated by append:
[0, 1, 2, 3, 4, 5]

Popping numbers from place number 1:
[0, 1, 2, 3, 4, 5]
1
[0, 2, 3, 4, 5]
2
[0, 3, 4, 5]
3
[0, 4, 5]
4
[0, 5]
5
[0]

Popping last item from place number 0:
[0]
0
[empty]

New list initiated by append:
[6, 5, 4, 3, 2, 1]

Popping numbers from middle of list:
[6, 5, 4, 3, 2, 1]
3
[6, 5, 4, 2, 1]
2
[6, 5, 4, 1]
1
[6, 5, 4]

Popping last items from place number 2, 1 and 0:
[6, 5, 4]
4
[6, 5]
5
[6]
6
[empty]

New list initiated by append:
[0, 1, 2, 3, 4, 5]

Removing numbers from end w/o popping:
[0, 1, 2, 3, 4, 5]
[0, 1, 2, 3, 4]
[0, 1, 2, 3]
[0, 1, 2]
[0, 1]
[0]
[empty]

New list initiated as empty:
[empty]

Appended 8 numbers to clean list:
[8, 7, 6, 5, 4, 3, 2, 1]

Inserting same numbers at every second step:
[8, 7, 6, 5, 4, 3, 2, 1]
[8, 8, 7, 6, 5, 4, 3, 2, 1]
[8, 8, 7, 7, 6, 5, 4, 3, 2, 1]
[8, 8, 7, 7, 6, 6, 5, 4, 3, 2, 1]
[8, 8, 7, 7, 6, 6, 5, 5, 4, 3, 2, 1]
[8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 2, 1]
[8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 1]
[8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1]
[8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]

Have changed item number 3 to 100 by indexing.Trying to retrieve same item by indexing: 
Item 3 is 100

Retrieving length from list by function as: 16 items (16 is correct)

We now conclude testing after having tested all methods

*/