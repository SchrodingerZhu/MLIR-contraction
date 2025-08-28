// Context Lookup computation in MLIR affine dialect with constant bounds
// CPU-feasible sizes with commonly used shapes for transformer models
// 
// \begin{table}[h]
//   \centering
//   \begin{tabular}{|c|c|l|}
//     \hline
//     Parameter & Value & Description \\
//     \hline
//     B & 4 & Batch size \\
//     H & 16 & Number of attention heads \\
//     S_q & 512 & Query sequence length \\
//     S_k & 512 & Key sequence length \\
//     D & 64 & Head dimension \\
//     \hline
//   \end{tabular}
//   \caption{Constant assignments for context lookup computation}
// \end{table}
// 
// Original C code:
// for (int b = 0; b < B; ++b)
//   for (int h = 0; h < H; ++h)
//     for (int q = 0; q < S_q; ++q)
//       for (int d = 0; d < D; ++d)
//         for (int k = 0; k < S_k; ++k)
//           O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d];

module attributes { "simulation.prologue" = "volatile double ARRAY_0[4][16][512][512], ARRAY_1[4][16][512][64], ARRAY_2[4][16][512][64];" } {
func.func @constant_context_lookup(%arg0: memref<4x16x512x512xf32>,     // Probability tensor P[4][16][512][512]
                                   %arg1: memref<4x16x512x64xf32>,     // Value tensor V[4][16][512][64]
                                   %arg2: memref<4x16x512x64xf32>) {   // Output tensor O[4][16][512][64]
  
  // Nested affine loops with constant bounds
  // This implements the context lookup: weighted sum of value vectors using attention probabilities
  affine.for %b = 0 to 4 {
    affine.for %h = 0 to 16 {
      affine.for %q = 0 to 512 {
        affine.for %d = 0 to 64 {
          affine.for %k = 0 to 512 {
            // Load probability value: P[b][h][q][k]
            %prob_val = affine.load %arg0[%b, %h, %q, %k] : memref<4x16x512x512xf32>
            
            // Load value vector element: V[b][h][k][d]
            %value_val = affine.load %arg1[%b, %h, %k, %d] : memref<4x16x512x64xf32>
            
            // Load current output value: O[b][h][q][d]
            // %output_val = affine.load %arg2[%b, %h, %q, %d] : memref<4x16x512x64xf32>
            %output_val = arith.constant 0.0 : f32
            
            // Compute multiplication: P[b][h][q][k] * V[b][h][k][d]
            %mul = arith.mulf %prob_val, %value_val : f32
            
            // Compute accumulation: O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d]
            %add = arith.addf %output_val, %mul : f32
            
            // Store result back to output tensor
            affine.store %add, %arg2[%b, %h, %q, %d] : memref<4x16x512x64xf32>
          }
        }
      }
    }
  }
  
  return
}
}
