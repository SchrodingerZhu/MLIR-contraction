module attributes {simulation.prologue = "volatile double ARRAY_0[32][64][128], ARRAY_1[32][128][96], ARRAY_2[32][64][96];"} {
  func.func @batched_gemm(%arg0: memref<32x64x128xf64>, %arg1: memref<32x128x96xf64>, %arg2: memref<32x64x96xf64>) {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 12 {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 8 {
              affine.for %arg8 = 0 to 8 {
                affine.for %arg9 = 0 to 8 {
                  affine.for %arg10 = 0 to 8 {
                    %0 = affine.load %arg0[%arg7 + %arg3 * 8, %arg8 + %arg4 * 8, %arg10 + %arg6 * 8] : memref<32x64x128xf64>
                    %1 = affine.load %arg1[%arg7 + %arg3 * 8, %arg10 + %arg6 * 8, %arg9 + %arg5 * 8] : memref<32x128x96xf64>
                    %2 = arith.mulf %0, %1 : f64
                    affine.store %2, %arg2[%arg7 + %arg3 * 8, %arg8 + %arg4 * 8, %arg9 + %arg5 * 8] : memref<32x64x96xf64>
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

