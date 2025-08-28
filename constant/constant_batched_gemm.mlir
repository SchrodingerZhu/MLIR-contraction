module {
    func.func @batched_gemm(%A: memref<32x64x80xf64>, %B: memref<32x80x96xf64>, %C: memref<32x64x96xf64>) {
        // C[p][i][j] = A[p][i][k] * B[p][k][j]
        affine.for %p = 0 to 32 {
            affine.for %i = 0 to 64 {
                affine.for %j = 0 to 96 {
                    affine.for %k = 0 to 80 {
                        %A_pik = affine.load %A[%p, %i, %k] : memref<32x64x80xf64>
                        %B_pkj = affine.load %B[%p, %k, %j] : memref<32x80x96xf64>
                        %prod = arith.mulf %A_pik, %B_pkj : f64
                        affine.store %prod, %C[%p, %i, %j] : memref<32x64x96xf64>
                    }
                }
            }
        }
        return
    }
}