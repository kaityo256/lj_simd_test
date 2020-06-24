all: a.out
CC=g++
CPPFLAGS= -O3 -Wall -Wextra -mavx2

a.out: lj.cpp
	$(CC) $(CPPFLAGS) $< -o $@
