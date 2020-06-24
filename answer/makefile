all: a.out
CC=g++
CPPFLAGS= -O3 -Wall -Wextra -mavx2 -std=c++11

a.out: lj.cpp
	$(CC) $(CPPFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -f a.out
