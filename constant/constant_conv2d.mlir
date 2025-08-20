// Convolution operation in MLIR affine dialect with constant bounds
// CPU-feasible sizes with commonly used shapes for CNN models
// 
// \begin{table}[h]
//   \centering
//   \begin{tabular}{|c|c|l|}
//     \hline
//     Parameter & Value & Description \\
//     \hline
//     N & 1 & Batch size \\
//     IC & 32 & Input channels \\
//     OC & 64 & Output channels \\
//     OH & 28 & Output height \\
//     OW & 28 & Output width \\
//     KH & 3 & Kernel height \\
//     KW & 3 & Kernel width \\
//     \hline
//   \end{tabular}
//   \caption{Constant assignments for 2D convolution}
// \end{table}

func.func @constant_conv2d(%arg0: memref<1x32x30x30xf32>,     // Input tensor I[1][32][30][30]
                           %arg1: memref<64x32x3x3xf32>,      // Weight tensor W[64][32][3][3]
                           %arg2: memref<1x64x28x28xf32>) {   // Output tensor O[1][64][28][28]
  
  // Nested affine loops with constant bounds
  affine.for %n = 0 to 1 {
    affine.for %oc = 0 to 64 {
      affine.for %ic = 0 to 32 {
        affine.for %oh = 0 to 28 {
          affine.for %ow = 0 to 28 {
            affine.for %kh = 0 to 3 {
              affine.for %kw = 0 to 3 {
                // Load input value: I[n][ic][oh + kh][ow + kw]
                %input_val = affine.load %arg0[%n, %ic, %oh + %kh, %ow + %kw] : memref<1x32x30x30xf32>
                
                // Load weight value: W[oc][ic][kh][kw]
                %weight_val = affine.load %arg1[%oc, %ic, %kh, %kw] : memref<64x32x3x3xf32>
                
                // Load current output value: O[n][oc][oh][ow]
                // %output_val = affine.load %arg2[%n, %oc, %oh, %ow] : memref<1x64x28x28xf32>
                %output_val = arith.constant 0.0 : f32
                
                // Compute multiplication
                %mul = arith.mulf %input_val, %weight_val : f32
                
                // Compute accumulation: O[n][oc][oh][ow] += I[n][ic][oh + kh][ow + kw] * W[oc][ic][kh][kw]
                %add = arith.addf %output_val, %mul : f32
                
                // Store result back to output
                affine.store %add, %arg2[%n, %oc, %oh, %ow] : memref<1x64x28x28xf32>
              }
            }
          }
        }
      }
    }
  }
  
  return
}
