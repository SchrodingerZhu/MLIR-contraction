module {
    func.func @matrix_matrix_contraction(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %rows_A: index, %cols_A: index, %cols_B: index) {
        // C[i][k] = A[i][j] * B[j][k]
        affine.for %i = 0 to %rows_A {
            affine.for %k = 0 to %cols_B {
                affine.for %j = 0 to %cols_A {
                    %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
                    %B_jk = affine.load %B[%j, %k] : memref<?x?xf64>
                    %prod = arith.mulf %A_ij, %B_jk : f64
                    affine.store %prod, %C[%i, %k] : memref<?x?xf64>
                }
            }
        }
        return
    }
}
