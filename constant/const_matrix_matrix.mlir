module {
    func.func @matrix_matrix_contraction(%A: memref<256x256xf64>, %B: memref<256x256xf64>, %C: memref<256x256xf64>) {
        // C[i][k] = A[i][j] * B[j][k]
        affine.for %i = 0 to 256 {
            affine.for %k = 0 to 256 {
                affine.for %j = 0 to 256 {
                    %A_ij = affine.load %A[%i, %j] : memref<256x256xf64>
                    %B_jk = affine.load %B[%j, %k] : memref<256x256xf64>
                    %prod = arith.mulf %A_ij, %B_jk : f64
                    affine.store %prod, %C[%i, %k] : memref<256x256xf64>
                }
            }
        }
        return
    }
}
