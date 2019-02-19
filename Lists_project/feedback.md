# Feedback on project 2

## Generelt
Dette ser veldig bra ut. Du har implementert all funksjonalitet som
oppgaven ber om, og det ser ut til at det aller meste
virker. Kommentarene under er for det meste småpirk og ting som kunne
vært gjort annerledes og kanskje bedre, uten at det du har gjort
nødvendigvis er feil.

Prosjektet er godkjent.


## Part 1

### 1a-b
Alt ser ut til å fungere her.

### 1c
I resize-funksjonen er det mer effektivt å bruke memcpy enn en
for-løkke til å kopiere data, men måten du har brukt
den som er spesifisert i oppgateksten så den er helt ok.

### 1d-h
Ser helt riktig ut, og er kodet temmelig likt som løsningsforslaget.

### 1i
Her sier vel oppgaven at du skal finne en ny kapasitet på formen capacity = 2^k,
og velge minst mulig k  slik at capacity > size. Din kode gjør vel noe litt
annet, men den virker og gir tilnærmet samme resultat.

### 1j
Ser bra ut

### Part 2
Ser helt riktig ut


### Part 3
Ser også helt riktig ut. For operasjonen "remove from back" oppgir du
"O(1) if initialized as a doubly linked list", men trenger vel strengt
tatt bare ha en tail ptr for at denne skal være O(1).

### Part 4
Også helt riktig, og pent kodet. Hvis jeg skal pirke så er det kanskje
litt unaturlig å ha "killed" som en variabel i CircLinkedList, siden den bare
brukes i det spesielle tilfellet med josephus-sequence.
I funksjonen josephus_sequence skriver du nøyaktig samme kode en gang før
while-løkka, og så en gang inne i while-løkka. Dette virker, men du kunne vel
gått rett inn i while-løkka?
