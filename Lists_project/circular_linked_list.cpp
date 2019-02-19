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

class CircLinkedList {
    /* List class using linked Nodes for storing data, also linking last node
    to first node 

    List can take new values at end (append), list can be indexed and the list 
    can be pretty printed. List has method for creating a josephus sequence.
    List uses dynamic memory allocation, and has destructor.*/

private:
    Node* head;
    Node* tail;
    bool killed;
    int size;

public:
    CircLinkedList() {
        head = nullptr;
        tail = nullptr;
        size = 0;
        killed = false;
    }

    CircLinkedList(int n) {
        head = nullptr;
        tail = nullptr;
        size = 0;
        killed = false;

        for (int i=1;i<=n;i++) {
            append(i);
        }
    }

    ~CircLinkedList() {
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

        while (current -> next != head) {
            next = current -> next;
            delete current;
            current = next;
            }
        delete current;
        }
    } 

        void append(int val) {
        if (size == 0) {
            head = new Node(val,head);
            tail = head;

        } else {
            tail -> next = new Node(val,head);
            tail = tail -> next;
        }

        size++;
    }
    
    int& get(int index) {
        if (size == 0) {
            throw out_of_range("IndexError");

        } else if (index == 0) {
            return head -> value;

        } else if (index == size - 1) {
            return tail -> value;
            
        } else {
            Node* current = head;
            int i = 0;

            while (i < index) {
                current = current -> next;
                i++;
            }
            return current -> value;
        }
    }

    int& operator[](int index) {
        return get(index);
    }

    void print() {
        if (size == 0) {
            cout << "[empty]" << endl;

        } else {
            Node* current = head;

            cout << "[";
            
            while (current -> next != head) {
                cout << current -> value;
                cout << ", ";
                current = current -> next;
            }

        cout << current ->value << ", ...]" << endl;
        }
    }

    vector<int> josephus_sequence(int k) {
        if (killed) {
            throw out_of_range("Josephus sequence already initiated. All killed.");

        } else {
            vector<int> josephus_list;
            Node* current = head;

            for (int i = 0; i < k - 2; i++) {
                current = current -> next;
            }

            Node* kill = current -> next;
            Node* splice = kill -> next;

            current -> next = splice;

            if (kill == head and size > 1) {
                head = splice;
            }

            if (kill == tail and size > 1) {
                tail = current;
            }

            if (size == 1) {
                head = nullptr;
                tail = nullptr;
            }

            josephus_list.push_back(kill->value);
            delete kill;

            size --;

            while (size > 0) {
                for (int i = 0; i < k - 1; i++) {
                    current = current -> next;
                }

                Node* kill = current -> next;
                Node* splice = kill -> next;

                current -> next = splice;

                if (kill == head and size > 1) {
                    head = splice;
                }

                if (kill == tail and size > 1) {
                    tail = current;
                }

                if (size == 1) {
                    head = nullptr;
                    tail = nullptr;
                }

                josephus_list.push_back(kill -> value);
                delete kill;                

                size --;
            }
            
        killed = true;
        return josephus_list;
        }
    }

};

int last_man_standing(int n, int k) {
    /* Calculates last man standing for a josephus problem with n participants
    where every kth person is killed, going around the list anew when the end
    is reached 

    Returns last man standing */

    CircLinkedList josephus_friends(n);
    vector<int> j_list = josephus_friends.josephus_sequence(k);

    /* Printing of whole sequence can be implemented: */
        // cout << "[";

        // for (int i=0;i<j_list.size()-1;i++) {
        //     cout << j_list[i];
        //     cout << ", ";
        // }

        // cout << j_list[j_list.size()-1] << "]" << endl;

    return j_list[j_list.size()-1];
}

void test_print() {
    /* Test checking print functionality */

    CircLinkedList clist;
    clist.append(0);
    clist.append(2);
    clist.append(4);

    cout << "Should be printed: \n[0, 2, 4, ...]\n";
    cout << "Is printed: " << endl;

    clist.print();
}

void test_get() {
    /* Test checking get functionality */

    CircLinkedList clist;
    clist.append(0);
    clist.append(2);
    clist.append(4);

    cout << "\nElement number 101 (circularly) is " << clist[101];
    cout << "\n(Correct number is 4)." << endl;
}

void test_overload_constructor() {
    /* Test checking overloading of constructor */

    CircLinkedList testlist(100);

    cout << "\nThis should be a print of a list of 100 elements 1 to 100:";
    cout << endl;
    testlist.print();
}

int main () {
    cout << "\nInitialize simple testing of class CircLinkedList:\n" << endl;
    test_print();
    test_get();
    test_overload_constructor();

    int k = 7;
    int n = 68;

    cout << "\nJosephus problem - solution:";
    cout << "\nThe last man standing out of " << n << " when every " << k;
    cout << " is killed is number " << last_man_standing(n,k) << ".\n" << endl;

    return 0;
}

/*
Output:

Initialize simple testing of class CircLinkedList:

Should be printed: 
[0, 2, 4, ...]
Is printed: 
[0, 2, 4, ...]

Element number 101 (circularly) is 4
(Correct number is 4).

This should be a print of a list of 100 elements 1 to 100:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, ...]

Josephus problem - solution:
The last man standing out of 68 when every 7 is killed is number 68.
*/