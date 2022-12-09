IDIR =include
CC=gcc
CFLAGS=-I $(IDIR) -fopenmp

ODIR=obj
LDIR=lib
SRCDIR=src

LIBS=-lm

_DEPS = layers.h definitions.h data.h network.h tests.h serialize.h utils.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = layers.o data.o network.o tests.o serialize.o utils.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $(SRCDIR)/$@.c $^ $(CFLAGS) $(LIBS)

demo: $(OBJ)
	$(CC) -o $@ $(SRCDIR)/$@.c $^ $(CFLAGS) $(LIBS)

test: $(OBJ)
	$(CC) -o $@ $(SRCDIR)/$@.c $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o main demo test *~ core $(INCDIR)/*~ 