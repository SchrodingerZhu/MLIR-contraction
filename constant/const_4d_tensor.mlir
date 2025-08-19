module {
    func.func @tensor4d_contraction(%A: memref<128x128x128x128xf64>, %B: memref<128x128xf64>, %C: memref<128x128xf64>) {
        // C[i][j] = A[i][k][l][j] * B[k][l]
        affine.for %i = 0 to 128 {
            affine.for %j = 0 to 128 {
                affine.for %k = 0 to 128 {
                    affine.for %l = 0 to 128 {
                        %A_iklj = affine.load %A[%i, %k, %l, %j] : memref<128x128x128x128xf64>
                        %B_kl = affine.load %B[%k, %l] : memref<128x128xf64>
                        %prod = arith.mulf %A_iklj, %B_kl : f64
                        affine.store %prod, %C[%i, %j] : memref<128x128xf64>
                    }
                }
            }
        }
        return
    }
}
