module {
    func.func @tensor3d_vector_contraction(%A: memref<?x?x?xf64>, %B: memref<?xf64>, %C: memref<?x?xf64>, %dim1: index, %dim2: index, %dim3: index) {
        // C[i][j] = A[i][j][k] * B[k]
        affine.for %i = 0 to %dim1 {
            affine.for %j = 0 to %dim2 {
                affine.for %k = 0 to %dim3 {
                    %A_ijk = affine.load %A[%i, %j, %k] : memref<?x?x?xf64>
                    %B_k = affine.load %B[%k] : memref<?xf64>
                    %prod = arith.mulf %A_ijk, %B_k : f64
                    affine.store %prod, %C[%i, %j] : memref<?x?xf64>
                }
            }
        }
        return
    }
}
