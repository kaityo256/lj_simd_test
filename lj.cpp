#include <cstdio>
#include <cstring>

enum { X,
       Y,
       Z };

const int ND = 3; // FCCの格子数
const int N = ND * ND * ND * 4;
double __attribute__((aligned(32))) q[N][4] = {};
double __attribute__((aligned(32))) p[N][4] = {};
double __attribute__((aligned(32))) p2[N][4] = {};
const double dt = 0.01;

void init(void) {
  for (int i = 0; i < N; i++) {
    p[i][X] = 0.0;
    p[i][Y] = 0.0;
    p[i][Z] = 0.0;
  }
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

// SIMD化していないシンプルな関数
void calc_force_simple(void) {
  for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
      double dx = q[j][X] - q[i][X];
      double dy = q[j][Y] - q[i][Y];
      double dz = q[j][Z] - q[i][Z];
      double r2 = dx * dx + dy * dy + dz * dz;
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      p[j][X] -= df * dx;
      p[j][Y] -= df * dy;
      p[j][Z] -= df * dz;
      p[i][X] += df * dx;
      p[i][Y] += df * dy;
      p[i][Z] += df * dz;
    }
  }
}

int main(void) {
  init();
  for (int i = 0; i < N; i++) {
    printf("%f %f %f\n", q[i][X], q[i][Y], q[i][Z]);
  }
  calc_force_simple();
  memcpy(p2, p, sizeof(p));
  init();
  calc_force_simple();
  int r = memcmp(p, p2, sizeof(p));
  printf("%d\n", r);
}
