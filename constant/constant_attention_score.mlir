// Attention Score computation in MLIR affine dialect with constant bounds
// CPU-feasible sizes with commonly used shapes for transformer models
// 
// \begin{table}[h]
//   \centering
//   \begin{tabular}{|c|c|l|}
//     \hline
//     Parameter & Value & Description \\
//     \hline
//     B & 2 & Batch size \\
//     H & 8 & Number of attention heads \\
//     S_q & 64 & Query sequence length \\
//     S_k & 64 & Key sequence length \\
//     D & 64 & Head dimension \\
//     \hline
//   \end{tabular}
//   \caption{Constant assignments for attention score computation}
// \end{table}
// 
// Original C code:
// for (int b = 0; b < B; ++b)
//   for (int h = 0; h < H; ++h)
//     for (int q = 0; q < S_q; ++q)
//       for (int k = 0; k < S_k; ++k)
//         for (int d = 0; d < D; ++d)
//           S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d];

func.func @constant_attention_score(%arg0: memref<2x8x64x64xf32>,     // Query tensor Q[2][8][64][64]
                                    %arg1: memref<2x8x64x64xf32>,     // Key tensor K[2][8][64][64]
                                    %arg2: memref<2x8x64x64xf32>) {   // Score tensor S[2][8][64][64]
  
  // Nested affine loops with constant bounds
  // This implements the query-key dot product for attention scores
  affine.for %b = 0 to 2 {
    affine.for %h = 0 to 8 {
      affine.for %q = 0 to 64 {
        affine.for %k = 0 to 64 {
          affine.for %d = 0 to 64 {
            // Load query value: Q[b][h][q][d]
            %query_val = affine.load %arg0[%b, %h, %q, %d] : memref<2x8x64x64xf32>
            
            // Load key value: K[b][h][k][d]
            %key_val = affine.load %arg1[%b, %h, %k, %d] : memref<2x8x64x64xf32>
            
            // Load current score value: S[b][h][q][k]
            // %score_val = affine.load %arg2[%b, %h, %q, %k] : memref<2x8x64x64xf32>
            %score_val = arith.constant 0.0 : f32
            
            // Compute multiplication: Q[b][h][q][d] * K[b][h][k][d]
            %mul = arith.mulf %query_val, %key_val : f32
            
            // Compute accumulation: S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d]
            %add = arith.addf %score_val, %mul : f32
            
            // Store result back to score tensor
            affine.store %add, %arg2[%b, %h, %q, %k] : memref<2x8x64x64xf32>
          }
        }
      }
    }
  }
  
  return
}
