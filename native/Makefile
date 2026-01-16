CC=cc
CFLAGS=-O3 -fPIC
LDFLAGS=-shared
LIBS=-llapacke -lopenblas

all: libsvd_shim.so

libsvd_shim.so: svd_shim.c
	$(CC) $(CFLAGS) $(LDFLAGS) svd_shim.c -o libsvd_shim.so $(LIBS)

clean:
	rm -f libsvd_shim.so

