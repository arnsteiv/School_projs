# IN1910 Prosjekt 2 (project2_arnsteiv)
**Arnstein Vestre** // **project2_arnsteiv**

Project 2 - IN1910 - Arnstein Vestre. The readme is used for explaining code compilation and answering part 3 and 4g (pop quiz).

## Code was compiled and run in the following way:

**Task 1: ArrayList**

arnsteinvestre @ project2_arnsteiv $ g++ -std=c++11 array_list.cpp -o "array_list"

arnsteinvestre @ project2_arnsteiv $ ./array_list

**Task 2: LinkedList**

arnsteinvestre @ project2_arnsteiv $ g++ -std=c++11 linked_list.cpp -o "linked_list"

arnsteinvestre @ project2_arnsteiv $ ./linked_list

**Task 4: CircularlyLinkedList**

arnsteinvestre @ project2_arnsteiv $ g++ -std=c++11 circular_linked_list.cpp -o "circular_linked_list"

arnsteinvestre @ project2_arnsteiv $ ./circular_linked_list

The programs were **originally compiled on local mac**, but **also successfully compiled and run on IFI linux machines**.

## Part 3: Comparing the ArrayList and LinkedList

We have the performance of the given tasks for the two different list implementations as following:

### Get element i by index

**ArrayList**: f(n) = O(1)

The ArrayList can retrieve the element in one operation through indexing. 

**LinkedList**: f(n) = O(i) (alternatively: f(n) = O(1) + search)

The LinkedList must search the list until the ith element by going from element
to element (O(i)), and then return it (O(1).

### Insert at front

**ArrayList**: f(n) = O(n) + resize

The ArrayList inserts at front by writing all elements 0 to n in positions 1 to n+1, and then inserting element 0. If the list capacity is spent, the list must be resized. 

**LinkedList**: f(n) = O(1)

The LinkedList redirects the head pointer and deletes the first element.

### Insert at back (aka append)

**ArrayList**: f(n) = O(1) + resize

The ArrayList appends by taking from its unused capcity and allocating a new
space in its array at the n+1th point. If capacity is spent, resize is also
needed.

**LinkedList**: f(n) = O(1) (O(n) if not initialized with a tail pointer)

The LinkedList can access the last node by the tail pointer, redirect this to 
a new node, and redirect the tail pointer. If a tail pointer is not included, 
iterating through the list is neccessary, and operations will be O(n).

### Insert into middle of list

**ArrayList**: f(n) = O(n-i) + resize (formally: O(n))

The ArrayList inserts in the middle by overwriting elements i+1 to n+1 with the
value from the i to nth entry, and then overwriting element i with the new 
value. If capacity is spent, resize is needed.

**LinkedList**: f(n) = O(i) (alternatively: f(n) = O(1) + search)

The LinkedList searchs the list until the ith element (O(i)), redirects the 
pointer in the i-1th element to a new element, and points this to the i+1th 
element (O(1)).

### Remove element from front

**ArrayList**: f(n) = O(n)

The ArrayList removes from the front by overwriting element 0 to n with element 1 to n-1.

**LinkedList**: f(n) = O(1)

The LinkedList redirects the head pointer to element 1 and deletes element 0.

### Remove element from back

**ArrayList**: f(n) = O(1)

The ArrayList removes from the back by subtracting from the size, thus making the
size inaccessable by the iterators in the class' methods.

**LinkedList**: f(n) = O(n) (O(1) if initialized as a doubly linked list)

The LinkedList iterates through the list to fund the n-1th pointer, redirects this to NULL, deletes the last element and redirects the tail pointer.

If the linked list is initialized as a doubly linked list, the n-1th element 
can be accessed by iterating from the back.

### Remove element from middle

**ArrayList**: f(n) = O(n-i) = O(n)

The ArrayList removes from the middle by overwriting elements i to n-1 with values from entries i+1 to n.

**LinkedList**: f(n) = O(i) (alternatively: f(n) = O(1) + search. n worst case. n/2 worst case if initialized as a doubly linked list)

The LinkedList searches through the list until element i (O(i)), then redirects pointer i-1 to element i+1 and deletes element i (O(1)).

### Print

**ArrayList**: f(n) = O(n)

**LinkedList**: f(n) = O(n)

Both lists need to iterate through the whole list to access each element and
send it to cout. The LinkedList has two operations for every iteration, the
ArrayList has 1, but in terms of Big Oh, they are in the same class (linear
growth).

### A comment on resizing

**ArrayLists** need to resize when adding an element, if the new size is exceeding the underlying storage capacity set aside for the list. To account for how this affects the Big Oh of the append and insert functions (the shrink_to_fit imple mented as part of remove-functions is negligable), we need to look at how this affects the operations on average. If we amortize the calculation (look at how the resize affects n operations of the sort and take the mean), we find that **in every occation, the Big Oh is reduced to O(n)**, so that the resize does not affect how the need for computational power scales.

## Part 4g: Pop quiz

We call the function last_man_standing(.) with n=68 and k=7. This 
returns **68**. If you are in a josephus kind of-party, **make sure to stand in place number 68** (the last place).


