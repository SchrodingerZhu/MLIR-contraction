module {
    func.func @tensor3d_vector_contraction(%A: memref<256x256x256xf64>, %B: memref<256xf64>, %C: memref<256x256xf64>) {
        // C[i][j] = A[i][j][k] * B[k]
        affine.for %i = 0 to 256 {
            affine.for %j = 0 to 256 {
                affine.for %k = 0 to 256 {
                    %A_ijk = affine.load %A[%i, %j, %k] : memref<256x256x256xf64>
                    %B_k = affine.load %B[%k] : memref<256xf64>
                    %prod = arith.mulf %A_ijk, %B_k : f64
                    affine.store %prod, %C[%i, %j] : memref<256x256xf64>
                }
            }
        }
        return
    }
}
