IDIR =include
CC=gcc
CFLAGS=-I $(IDIR) -fopenmp

ODIR=obj
LDIR=lib
SRCDIR=src

LIBS=-lm

_DEPS = lib.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = lib.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $(SRCDIR)/$@.c $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o main *~ core $(INCDIR)/*~ 