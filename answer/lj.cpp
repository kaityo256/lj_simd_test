#include <chrono>
#include <cstdio>
#include <cstring>
#include <x86intrin.h>
#include <iostream>
#include <fstream>
#include <sstream>

enum { X,
       Y,
       Z };

const int ND = 10;                                 // FCCの格子数
const int N = ND * ND * ND * 4;                    //全粒指数
double __attribute__((aligned(32))) q[N][4] = {};  //座標
double __attribute__((aligned(32))) p[N][4] = {};  //運動量
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
  for (int i = 0; i < N-1; i++) {
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
  const __m256d vc24 = _mm256_set_pd(24 * dt, 24 * dt, 24 * dt, 24 * dt);
  const __m256d vc48 = _mm256_set_pd(48 * dt, 48 * dt, 48 * dt, 48 * dt);

  for (int i = 0; i < N-1; i++) {
    __m256d vqi = _mm256_load_pd((double *)(q + i));
    double qix = vqi[X];
    double qiy = vqi[Y];
    double qiz = vqi[Z];

    __m256d vpi = _mm256_load_pd((double *)(p + i));
    int j = i + 1;
    for (; j + 3 < N; j += 4) {
      int j_0 = j;
      int j_1 = j + 1;
      int j_2 = j + 2;
      int j_3 = j + 3;

      __m256d vqj_0 = _mm256_load_pd((double *)(q + j_0));
      __m256d vqj_1 = _mm256_load_pd((double *)(q + j_1));
      __m256d vqj_2 = _mm256_load_pd((double *)(q + j_2));
      __m256d vqj_3 = _mm256_load_pd((double *)(q + j_3));

      __m256d vdr_0 = vqj_0 - vqi;
      __m256d vdr_1 = vqj_1 - vqi;
      __m256d vdr_2 = vqj_2 - vqi;
      __m256d vdr_3 = vqj_3 - vqi;

      __m256d tmp0 = _mm256_unpacklo_pd(vdr_0, vdr_1);
      __m256d tmp1 = _mm256_unpackhi_pd(vdr_0, vdr_1);
      __m256d tmp2 = _mm256_unpacklo_pd(vdr_2, vdr_3);
      __m256d tmp3 = _mm256_unpackhi_pd(vdr_2, vdr_3);

      __m256d vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      __m256d vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      __m256d vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

      __m256d vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      __m256d vr6 = vr2 * vr2 * vr2;
      __m256d vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      __m256d vdf_0 = _mm256_permute4x64_pd(vdf, 0);
      __m256d vdf_1 = _mm256_permute4x64_pd(vdf, 85);
      __m256d vdf_2 = _mm256_permute4x64_pd(vdf, 170);
      __m256d vdf_3 = _mm256_permute4x64_pd(vdf, 255);
 
      __m256d vpj_0 = _mm256_load_pd((double *)(p + j_0));
      __m256d vpj_1 = _mm256_load_pd((double *)(p + j_1));
      __m256d vpj_2 = _mm256_load_pd((double *)(p + j_2));
      __m256d vpj_3 = _mm256_load_pd((double *)(p + j_3));
      vpj_0 -= vdf_0 * vdr_0;
      vpj_1 -= vdf_1 * vdr_1;
      vpj_2 -= vdf_2 * vdr_2;
      vpj_3 -= vdf_3 * vdr_3;
      _mm256_store_pd(&(p[j_0][0]), vpj_0);
      _mm256_store_pd(&(p[j_1][0]), vpj_1);
      _mm256_store_pd(&(p[j_2][0]), vpj_2);
      _mm256_store_pd(&(p[j_3][0]), vpj_3);
      vpi += vdf_0 * vdr_0;
      vpi += vdf_1 * vdr_1;
      vpi += vdf_2 * vdr_2;
      vpi += vdf_3 * vdr_3;
    }
    _mm256_store_pd(&(p[i][0]), vpi);

    for (; j < N; j++) {
      double dx = q[j][X] - qix;
      double dy = q[j][Y] - qiy;
      double dz = q[j][Z] - qiz;
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


// まず、ループを4倍展開した関数
void calc_force_unrolled(void) {
  for (int i = 0; i < N-1; i++) {
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
    int j = i + 1;
    for (; j + 3 < N; j += 4) {
      int j_0 = j;
      int j_1 = j + 1;
      int j_2 = j + 2;
      int j_3 = j + 3;

      double dx_0 = q[j_0][X] - qix;
      double dy_0 = q[j_0][Y] - qiy;
      double dz_0 = q[j_0][Z] - qiz;
      double r2_0 = dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0;
      double r6_0 = r2_0 * r2_0 * r2_0;
      double df_0 = (24.0 * r6_0 - 48.0) / (r6_0 * r6_0 * r2_0) * dt;
      p[j_0][X] -= df_0 * dx_0;
      p[j_0][Y] -= df_0 * dy_0;
      p[j_0][Z] -= df_0 * dz_0;
      pix += df_0 * dx_0;
      piy += df_0 * dy_0;
      piz += df_0 * dz_0;

      double dx_1 = q[j_1][X] - qix;
      double dy_1 = q[j_1][Y] - qiy;
      double dz_1 = q[j_1][Z] - qiz;
      double r2_1 = dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1;
      double r6_1 = r2_1 * r2_1 * r2_1;
      double df_1 = (24.0 * r6_1 - 48.0) / (r6_1 * r6_1 * r2_1) * dt;
      p[j_1][X] -= df_1 * dx_1;
      p[j_1][Y] -= df_1 * dy_1;
      p[j_1][Z] -= df_1 * dz_1;
      pix += df_1 * dx_1;
      piy += df_1 * dy_1;
      piz += df_1 * dz_1;

      double dx_2 = q[j_2][X] - qix;
      double dy_2 = q[j_2][Y] - qiy;
      double dz_2 = q[j_2][Z] - qiz;
      double r2_2 = dx_2 * dx_2 + dy_2 * dy_2 + dz_2 * dz_2;
      double r6_2 = r2_2 * r2_2 * r2_2;
      double df_2 = (24.0 * r6_2 - 48.0) / (r6_2 * r6_2 * r2_2) * dt;
      p[j_2][X] -= df_2 * dx_2;
      p[j_2][Y] -= df_2 * dy_2;
      p[j_2][Z] -= df_2 * dz_2;
      pix += df_2 * dx_2;
      piy += df_2 * dy_2;
      piz += df_2 * dz_2;

      double dx_3 = q[j_3][X] - qix;
      double dy_3 = q[j_3][Y] - qiy;
      double dz_3 = q[j_3][Z] - qiz;
      double r2_3 = dx_3 * dx_3 + dy_3 * dy_3 + dz_3 * dz_3;
      double r6_3 = r2_3 * r2_3 * r2_3;
      double df_3 = (24.0 * r6_3 - 48.0) / (r6_3 * r6_3 * r2_3) * dt;
      p[j_3][X] -= df_3 * dx_3;
      p[j_3][Y] -= df_3 * dy_3;
      p[j_3][Z] -= df_3 * dz_3;
      pix += df_3 * dx_3;
      piy += df_3 * dy_3;
      piz += df_3 * dz_3;
    }

    for (; j < N; j++) {
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
  printf("N=%d, %s \t %ld [ms]\n", particle_number, name, elapsed);
}

/*
 運動量を文字列にして返す関数
 後で結果をチェックするのに使う
 */
std::string p_to_str(void){
  std::stringstream ss;  
  for(int i=0;i<N;i++){
    ss << i << " ";
    ss << p[i][X] << " ";
    ss << p[i][Y] << " ";
    ss << p[i][Z] << std::endl;
  }
  return ss.str();
}


int main(void) {
  // まずSIMD化していない関数の実行時間を測定し、結果を文字列として保存する
  init();
  measure(calc_force_simple, "simple", N);
  std::string simple = p_to_str();

  // 次にSIMD化した関数の実行時間を測定し、結果を文字列として保存する
  init();
  measure(calc_force_simd, "simd", N);
  std::string simd = p_to_str();

  if (simple == simd) {
    // 二つの結果が一致すればOK
    printf("Check OK\n");
  } else {
    // そうでなければ、なにかが間違っている。
    printf("Check Failed\n");
  }
}