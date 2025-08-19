module {
    func.func @batched_gemm(%A: memref<128x128x128xf64>, %B: memref<128x128x128xf64>, %C: memref<128x128x128xf64>) {
        // C[b][i][j] = A[b][i][k] * B[b][k][j]
        affine.for %b = 0 to 128 {
            affine.for %i = 0 to 128 {
                affine.for %j = 0 to 128 {
                    affine.for %k = 0 to 128 {
                        %A_bik = affine.load %A[%b, %i, %k] : memref<128x128x128xf64>
                        %B_bkj = affine.load %B[%b, %k, %j] : memref<128x128x128xf64>
                        %prod = arith.mulf %A_bik, %B_bkj : f64
                        affine.store %prod, %C[%b, %i, %j] : memref<128x128x128xf64>
                    }
                }
            }
        }
        return
    }
}
