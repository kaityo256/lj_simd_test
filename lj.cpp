#include <chrono>
#include <cstdio>
#include <cstring>
#include <x86intrin.h>

enum { X,
       Y,
       Z };

const int ND = 10;                                 // FCCの格子数
const int N = ND * ND * ND * 4;                    //全粒指数
double __attribute__((aligned(32))) q[N][4] = {};  //座標
double __attribute__((aligned(32))) p[N][4] = {};  //運動量
double __attribute__((aligned(32))) p2[N][4] = {}; //運動量(保存用) 後でSIMD版と結果を確認するのに用いる
const double dt = 0.01;

/*
 256bit浮動小数点レジスタの中身を表示する関数
 4つの倍精度浮動小数点数(64bit)をまとめたものになっているので、それをバラす
*/
void print256d(__m256d x) {
  printf("%f %f %f %f\n", x[3], x[2], x[1], x[0]);
}

/*
　初期化をする関数
  運動量をすべて0クリア
  座標はFCCに組む
 */
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
    // i粒子の座標と運動量を受け取っておく (内側のループでiは変化しないから)
    double qix = q[i][X];
    double qiy = q[i][Y];
    double qiz = q[i][Z];
    double pix = p[i][X];
    double piy = p[i][Y];
    double piz = p[i][Z];
    for (int j = i + 1; j < N; j++) {
      double dx = q[j][X] - qix;
      double dy = q[j][Y] - qiy;
      double dz = q[j][Z] - qiz;
      double r2 = dx * dx + dy * dy + dz * dz;
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      p[j][X] -= df * dx;
      p[j][Y] -= df * dy;
      p[j][Z] -= df * dz;
      pix += df * dx;
      piy += df * dy;
      piz += df * dz;
    }
    // 内側のループで積算したi粒子への力積を書き戻す
    p[i][X] = pix;
    p[i][Y] = piy;
    p[i][Z] = piz;
  }
}

// SIMD化した関数
void calc_force_simd(void) {
  for (int i = 0; i < N - 1; i++) {
    // i粒子の座標と運動量を受け取っておく (内側のループでiは変化しないから)

    // 4成分をまとめてロードする
    __m256d vqi = _mm256_load_pd((double *)(q + i));

    // まとめたまま計算した方が早いが、デバッグのために各成分にバラす (pも同様)
    double qix = vqi[X];
    double qiy = vqi[Y];
    double qiz = vqi[Z];

    /*
    上記のコードは、以下のコードをSIMD化したもの
    double qix = q[i][X];
    double qiy = q[i][Y];
    double qiz = q[i][Z];
    */

    __m256d vpi = _mm256_load_pd((double *)(p + i));
    double pix = vpi[X];
    double piy = vpi[Y];
    double piz = vpi[Z];

    /*
    上記のコードは、以下のコードをSIMD化したもの
    double pix = p[i][X];
    double piy = p[i][Y];
    double piz = p[i][Z];
    */

    for (int j = i + 1; j < N; j++) {
      double dx = q[j][X] - qix;
      double dy = q[j][Y] - qiy;
      double dz = q[j][Z] - qiz;
      double r2 = dx * dx + dy * dy + dz * dz;
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      p[j][X] -= df * dx;
      p[j][Y] -= df * dy;
      p[j][Z] -= df * dz;
      pix += df * dx;
      piy += df * dy;
      piz += df * dz;
    }
    // 内側のループで積算したi粒子への力積を書き戻す
    vpi = _mm256_set_pd(0.0, piz, piy, pix);
    _mm256_store_pd(&(p[i][0]), vpi);

    /*
    上記のコードは、以下のコードをSIMD化したもの
    p[i][X] = pix;
    p[i][Y] = piy;
    p[i][Z] = piz;
    */
  }
}

// インテルコンパイラのループ交換最適化阻害のためのダミー変数
int sum = 0;

/*
 受け取った関数を100回繰り返し実行して、実行時間を計測する
 */
void measure(void (*pfunc)(), const char *name, int particle_number) {
  const auto s = std::chrono::system_clock::now();
  const int LOOP = 100;
  for (int i = 0; i < LOOP; i++) {
    sum++; // ループ交換阻害
    pfunc();
  }
  const auto e = std::chrono::system_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
  printf("N=%d, %s \t %lld [ms]\n", particle_number, name, elapsed);
}

int main(void) {
  init();
  measure(calc_force_simple, "simple", N);
  // 計算した運動量pを比較用にp2に保存しておく
  memcpy(p2, p, sizeof(p));
  init();
  measure(calc_force_simd, "simd", N);
  int r = memcmp(p, p2, sizeof(p));

  if (r == 0) {
    // 二つの結果が一致すればOK
    printf("Check OK\n");
  } else {
    // そうでなければ、なにかが間違っている。
    printf("Check Failed\n");
  }
}
