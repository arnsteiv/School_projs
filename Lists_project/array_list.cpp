#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

bool is_prime(int n) {
    /* Simple optimized function for checking if int is prime */

    // Check for integers 1-3
    if (n == 1) {
        return false;
    } 
    if (n == 2) {
        return true;
    } 
    if (n == 3) {
        return true;
    } 

    // Check for even number
    if (n % 2 == 0) {
        return false;
    }

    // Check for divisibility by 3
    if (n % 3 == 0) {
        return false;
    }

    // Check all non-even numbers 3-sqrt(n) for divisibility
    for (int i=3;i*i<=n;i+=2) {
        if (((i-1) % 6 != 0) && ((i+1) % 6 != 0)) {
            ;
        }
        else if (n % i == 0) {
            return false;
        }
    }
    return true;
}

class ArrayList {
    /* List class that uses dynamic allocation of arrays to store values 

    List can take new values at end (append), values can be inserted, removed,
    popped and popped from end without index. List can be indexed, length of
    list can be retrieved, and the list can be pretty printed. List uses 
    dynamic memory allocation, and has destructor.*/

private:
    int *data;
    int capacity;

    void resize() {
        capacity *= 2;
        int *tmp = new int[capacity];

        for (int i=0; i<size;i++) {
            tmp[i] = data[i];
        }

        delete[] data;
        data = tmp;

    }

public:
    int size;

    ArrayList () {
        capacity = 2;
        data = new int[capacity];
        size = 0;
    }

    ArrayList(vector<int> inlist) {
        capacity = 2;
        size = inlist.size();

        while (capacity < size) {
            capacity *= 2;
        }

        data = new int[capacity];

        for (int i=0;i<size;i++) {
            data[i] = inlist[i];
        }
    }

    ~ArrayList() {
        delete[] data;
    }


    void shrink_to_fit() {
        while (capacity > 2*size and capacity > 2) {
            capacity /= 2;
        }

        int *tmp = new int[capacity];

        for (int i=0; i<size;i++) {
            tmp[i] = data[i];
        }

        delete[] data;
        data = tmp;
    }

    int length() {
        return size;
    }

    void append(int n) {
        if (size >= capacity) {
            resize();
        }

        data[size] = n;
        size += 1;
    }

    int& operator[](int i) {
        if (i >= size or i < 0) {
            throw out_of_range("IndexError");

        } else {
            return data[i];
        }
    }

    void insert(int val, int index) {
        if (size >= capacity) {
            resize();
        }

        if (index > size or index < 0) {
            throw out_of_range("IndexError");

        } else if (index == size) {
            append(val);

        } else {
            for (int i=size;i>index;i--) {
                data[i] = data[i-1];
            }

            data[index] = val;
        }
        size ++;
    }

    void remove(int n) {
        if (n >= size or n < 0) {
            throw out_of_range("IndexError");

        } else if (n == size - 1) {
            ;

        } else {
            for (int i=n;i<size;i++) {
                data[i] = data[i+1];
            }
        }
        size --;

        if (size < 0.25*capacity) {
            shrink_to_fit();
        }
    }

    int pop(int i) {
        int ret = data[i];
        remove(i);
        return ret;
    }

    int pop() {
        return pop(size-1);
    }

    void print() {
        if (size == 0) {
            cout << "[empty]" << endl;

        } else {
            cout << "[";

            for (int i=0;i<size-1;i++) {
                cout << data[i];
                cout << ", ";
            }

            cout << data[size-1];
            cout << "]" << endl;
        }
    }

    int get_capacity() {
        return capacity;
    }
};

void test_ArrayList_append_print() {
    /* Tests print function with primes function */

    cout << "\nTest should return: \n[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n\n";
    cout << "Returns:" << endl;
    ArrayList primes;
    int i = 1;
    while (primes.length()<10) {
        if (is_prime(i)) {
            primes.append(i);
        }
        i++;
    }
    primes.print();
}

void test_shrink_to_fit() {
    /* Test function for shrink_to_fit method (PS: now implemented in all
    relevant methods) */

    cout << "\nTesting Shrink-to-fit method:" << endl;

    ArrayList test({3,6,8,1,-4,3,6,8,1,-4});

    cout << "\nInitialized list from vector:" << endl;
    test.print();

    cout << "\nList has " << test.length() << " elements, and capacity ";
    cout << test.get_capacity() << endl;

    cout << "\nWe will now pop 7 items:" << endl;

    for (int i=0;i<7;i++) {
        test.pop();
        test.print();
    }

    cout << "\nWe now have a list of " << test.length() << " elements. ";
    cout << "List has capacity ";
    cout << test.get_capacity() << "." << endl;

    int test_cap = test.get_capacity();
    test.shrink_to_fit();

    cout << "\nWe now run a call for shrink-to-fit. New capacity is ";
    cout << test.get_capacity() << ".\n" << endl;

    if (test_cap == test.get_capacity()) {
        cout << "We see that automatic shrink-to-fit is successfully ";
        cout << "implemented in class\n" << endl;

    } else {
        cout << "Automatic shrink-to-fit is not implemented in class\n" << endl;
    }
}

int main () {
    cout << "\nInitializing simple testing of ArrayList class" << endl;
    test_ArrayList_append_print();
    test_shrink_to_fit();

    return 0;
}

/*
Output:


Initializing simple testing of ArrayList class

Test should return: 
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

Returns:
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

Testing Shrink-to-fit method:

Initialized list from vector:
[3, 6, 8, 1, -4, 3, 6, 8, 1, -4]

List has 10 elements, and capacity 16

We will now pop 7 items:
[3, 6, 8, 1, -4, 3, 6, 8, 1]
[3, 6, 8, 1, -4, 3, 6, 8]
[3, 6, 8, 1, -4, 3, 6]
[3, 6, 8, 1, -4, 3]
[3, 6, 8, 1, -4]
[3, 6, 8, 1]
[3, 6, 8]

We now have a list of 3 elements. List has capacity 4.

We now run a call for shrink-to-fit. New capacity is 4.

We see that automatic shrink-to-fit is successfully implemented in class

*/
