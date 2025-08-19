module {
    func.func @batched_gemm(%A: memref<?x?x?xf64>, %B: memref<?x?x?xf64>, %C: memref<?x?x?xf64>, %batch_size: index, %M: index, %N: index, %K: index) {
        // C[b][i][j] = A[b][i][k] * B[b][k][j]
        affine.for %b = 0 to %batch_size {
            affine.for %i = 0 to %M {
                affine.for %j = 0 to %N {
                    affine.for %k = 0 to %K {
                        %A_bik = affine.load %A[%b, %i, %k] : memref<?x?x?xf64>
                        %B_bkj = affine.load %B[%b, %k, %j] : memref<?x?x?xf64>
                        %prod = arith.mulf %A_bik, %B_bkj : f64
                        affine.store %prod, %C[%b, %i, %j] : memref<?x?x?xf64>
                    }
                }
            }
        }
        return
    }
}
