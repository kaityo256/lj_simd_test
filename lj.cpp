#include <cstdio>

enum { X,
       Y,
       Z };

const int ND = 3; // FCCの格子数
const int N = ND * ND * ND * 4;
double __attribute__((aligned(32))) q[N][4] = {};
double __attribute__((aligned(32))) p[N][4] = {};

void init(double q[N][4]) {
  for (int iz = 0; iz < ND; iz++) {
    for (int iy = 0; iy < ND; iy++) {
      for (int ix = 0; ix < ND; ix++) {
        int i = (ix + iy * ND + iz * ND * ND) * 4;
        q[i][X] = ix;
        q[i][Y] = iy;
        q[i][Z] = iz;
        q[i + 1][X] = ix + 0.5;
        q[i + 1][Y] = iy;
        q[i + 1][Z] = iz;
        q[i + 2][X] = ix;
        q[i + 2][Y] = iy + 0.5;
        q[i + 2][Z] = iz;
        q[i + 3][X] = ix;
        q[i + 3][Y] = iy;
        q[i + 3][Z] = iz + 0.5;
      }
    }
  }
}

int main(void) {
  init(q);
  for (int i = 0; i < N; i++) {
    printf("%f %f %f\n", q[i][X], q[i][Y], q[i][Z]);
  }
}
