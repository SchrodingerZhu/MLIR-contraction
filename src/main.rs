use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Addr([usize; 8]);

impl Addr {
    fn new(data: impl IntoIterator<Item = usize>) -> Self {
        let mut addr = Self([0; 8]);
        for (i, v) in addr.0.iter_mut().zip(data.into_iter()) {
            *i = v;
        }
        addr
    }
}

struct Cache {
    lastest_access: FxHashMap<Addr, usize>,
    ri_distogram: FxHashMap<usize, usize>,
    timer: usize,
    block_size: usize,
    start_recording: bool,
}

impl Cache {
    fn new(block_size: usize) -> Self {
        Self {
            lastest_access: FxHashMap::default(),
            ri_distogram: FxHashMap::default(),
            timer: 0,
            block_size,
            start_recording: false,
        }
    }

    fn access(&mut self, addr: impl Iterator<Item = usize> + ExactSizeIterator) {
        let last_pos = addr.len() - 1;
        let mut addr = Addr::new(addr);
        addr.0[last_pos] /= self.block_size;
        match self.lastest_access.entry(addr) {
            Entry::Occupied(mut entry) => {
                if self.start_recording {
                    self.ri_distogram
                        .entry(self.timer - entry.get())
                        .and_modify(|v| *v += 1)
                        .or_insert(1);
                }
                entry.insert(self.timer);
            }
            Entry::Vacant(entry) => {
                entry.insert(self.timer);
            }
        }
        self.timer += 1;
    }
}

#[allow(non_snake_case)]
fn attention_context(cache: &mut Cache, B: usize, H: usize, S_q: usize, D: usize, S_k: usize) {
    const O: usize = 0;
    const P: usize = 1;
    const V: usize = 2;

    for b in 0..B {
        for h in 0..H {
            for q in 0..S_q {
                for d in 0..D {
                    for k in 0..S_k {
                        cache.access([P, b, h, q, k].into_iter());
                        cache.access([V, b, h, k, d].into_iter());
                        cache.access([O, b, h, q, d].into_iter());
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn attention_score(cache: &mut Cache, B: usize, H: usize, S_q: usize, S_k: usize, D: usize) {
    const Q: usize = 0;
    const K: usize = 1;
    const S: usize = 2;

    for b in 0..B {
        for h in 0..H {
            for q in 0..S_q {
                for k in 0..S_k {
                    for d in 0..D {
                        cache.access([Q, b, h, q, d].into_iter());
                        cache.access([K, b, h, k, d].into_iter());
                        cache.access([S, b, h, q, k].into_iter());
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn conv2d(
    cache: &mut Cache,
    N: usize,
    OC: usize,
    IC: usize,
    OH: usize,
    OW: usize,
    KH: usize,
    KW: usize,
) {
    const I: usize = 0;
    const W: usize = 1;
    const O: usize = 2;

    for n in 0..N {
        for oc in 0..OC {
            for ic in 0..IC {
                for oh in 0..OH {
                    for ow in 0..OW {
                        for kh in 0..KH {
                            for kw in 0..KW {
                                cache.access([I, n, ic, oh + kh, ow + kw].into_iter());
                                cache.access([W, oc, ic, kh, kw].into_iter());
                                cache.access([O, n, oc, oh, ow].into_iter());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn depthwise_conv2d(
    cache: &mut Cache,
    N: usize,
    C: usize,
    OH: usize,
    OW: usize,
    KH: usize,
    KW: usize,
) {
    const I: usize = 0;
    const W: usize = 1;
    const O: usize = 2;

    for n in 0..N {
        for c in 0..C {
            for oh in 0..OH {
                for ow in 0..OW {
                    for kh in 0..KH {
                        for kw in 0..KW {
                            cache.access([I, n, c, oh + kh, ow + kw].into_iter());
                            cache.access([W, c, kh, kw].into_iter());
                            cache.access([O, n, c, oh, ow].into_iter());
                        }
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn pooling(cache: &mut Cache, N: usize, C: usize, OH: usize, OW: usize, KH: usize, KW: usize) {
    const I: usize = 0;
    const O: usize = 1;

    for n in 0..N {
        for c in 0..C {
            for oh in 0..OH {
                for ow in 0..OW {
                    for kh in 0..KH {
                        for kw in 0..KW {
                            cache.access([I, n, c, oh + kh, ow + kw].into_iter());
                            cache.access([O, n, c, oh, ow].into_iter());
                        }
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn rowwise_softmax_max(cache: &mut Cache, B: usize, H: usize, S_q: usize, S_k: usize) {
    const S: usize = 0;
    const M: usize = 1;

    for b in 0..B {
        for h in 0..H {
            for q in 0..S_q {
                for k in 0..S_k {
                    cache.access([S, b, h, q, k].into_iter());
                    cache.access([M, b, h, q].into_iter());
                }
            }
        }
    }
}
#[allow(non_snake_case)]
fn tensor3d_vector_contraction(cache: &mut Cache, M: usize, N: usize, K: usize) {
    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    for i in 0..M {
        for j in 0..N {
            for k in 0..K {
                cache.access([A, i, j, k].into_iter());
                cache.access([B, k].into_iter());
                cache.access([C, i, j].into_iter());
            }
        }
    }
}

#[allow(non_snake_case)]
fn tensor4d_contraction(cache: &mut Cache, M: usize, N: usize, K: usize, L: usize) {
    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    for i in 0..M {
        for j in 0..N {
            for k in 0..K {
                for l in 0..L {
                    cache.access([A, i, k, l, j].into_iter());
                    cache.access([B, k, l].into_iter());
                    cache.access([C, i, j].into_iter());
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn matrix_matrix_contraction(cache: &mut Cache, M: usize, N: usize, K: usize) {
    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    for i in 0..M {
        for k in 0..N {
            for j in 0..K {
                cache.access([A, i, j].into_iter());
                cache.access([B, j, k].into_iter());
                cache.access([C, i, k].into_iter());
            }
        }
    }
}

#[allow(non_snake_case)]
fn matrix_vector_contraction(cache: &mut Cache, M: usize, N: usize) {
    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    for i in 0..M {
        for j in 0..N {
            cache.access([A, i, j].into_iter());
            cache.access([B, j].into_iter());
            cache.access([C, i].into_iter());
        }
    }
}

#[allow(non_snake_case)]
fn batched_gemm(cache: &mut Cache, P: usize, M: usize, N: usize, K: usize) {
    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    for p in 0..P {
        for i in 0..M {
            for j in 0..N {
                for k in 0..K {
                    cache.access([A, p, i, k].into_iter());
                    cache.access([B, p, k, j].into_iter());
                    cache.access([C, p, i, j].into_iter());
                }
            }
        }
    }
}

#[allow(unused)]
fn main() {
    use indicatif::ParallelProgressIterator;
    use rayon::prelude::*;
    tracing_subscriber::fmt::init();

    let attention_context_runner = |cache: &mut Cache| {
        attention_context(cache, 2, 8, 64, 64, 64);
    };

    let attention_score_runner = |cache: &mut Cache| {
        attention_score(cache, 2, 8, 64, 64, 64);
    };

    let conv2d_runner = |cache: &mut Cache| {
        conv2d(cache, 1, 64, 32, 28, 28, 3, 3);
    };

    let depthwise_conv2d_runner = |cache: &mut Cache| {
        depthwise_conv2d(cache, 1, 128, 56, 56, 3, 3);
    };

    let pooling_runner = |cache: &mut Cache| {
        pooling(cache, 1, 64, 14, 14, 2, 2);
    };

    let rowwise_softmax_max_runner = |cache: &mut Cache| {
        rowwise_softmax_max(cache, 2, 8, 64, 64);
    };

    let tensor3d_vector_contraction_runner = |cache: &mut Cache| {
        tensor3d_vector_contraction(cache, 96, 64, 48);
    };

    let tensor4d_contraction_runner = |cache: &mut Cache| {
        tensor4d_contraction(cache, 80, 64, 48, 32);
    };

    let matrix_matrix_contraction_runner = |cache: &mut Cache| {
        matrix_matrix_contraction(cache, 256, 192, 160);
    };

    let matrix_vector_contraction_runner = |cache: &mut Cache| {
        matrix_vector_contraction(cache, 3840, 4096);
    };

    let batched_gemm_runner = |cache: &mut Cache| {
        batched_gemm(cache, 32, 64, 96, 80);
    };


    let test_cases: &[(&str, fn(&mut Cache), usize)] = &[
        ("Attention Context", attention_context_runner, 1),
        ("Attention Context", attention_context_runner, 8),
        ("Attention Score", attention_score_runner, 1),
        ("Attention Score", attention_score_runner, 8),
        ("Conv2D", conv2d_runner, 1),
        ("Conv2D", conv2d_runner, 8),
        ("Depthwise Conv2D", depthwise_conv2d_runner, 1),
        ("Depthwise Conv2D", depthwise_conv2d_runner, 8),
        ("Pooling", pooling_runner, 1),
        ("Pooling", pooling_runner, 8),
        ("Rowwise Softmax Max", rowwise_softmax_max_runner, 1),
        ("Rowwise Softmax Max", rowwise_softmax_max_runner, 8),
        ("Tensor3D Vector Contraction", tensor3d_vector_contraction_runner, 1),
        ("Tensor3D Vector Contraction", tensor3d_vector_contraction_runner, 8),
        ("Tensor4D Contraction", tensor4d_contraction_runner, 1),
        ("Tensor4D Contraction", tensor4d_contraction_runner, 8),
        ("Matrix Matrix Contraction", matrix_matrix_contraction_runner, 1),
        ("Matrix Matrix Contraction", matrix_matrix_contraction_runner, 8),
        ("Matrix Vector Contraction", matrix_vector_contraction_runner, 1),
        ("Matrix Vector Contraction", matrix_vector_contraction_runner, 8),
        ("Batched GEMM", batched_gemm_runner, 1),
        ("Batched GEMM", batched_gemm_runner, 8),
    ];  

    test_cases
        .par_iter()
        .progress()
        .for_each(|((name, runner, block_size))| {
            let mut cache = Cache::new(*block_size);
            runner(&mut cache);
            cache.start_recording = true;
            runner(&mut cache);
            let mut histogram = cache
                .ri_distogram
                .iter()
                .map(|(k, v)| (*k, *v as f64))
                .collect::<Vec<_>>();
            let total_access = histogram.iter().map(|(_, v)| *v).sum::<f64>();
            histogram.iter_mut().for_each(|(_, v)| *v /= total_access);
            histogram.sort_by(|a, b| a.0.cmp(&b.0));
            tracing::trace!("histogram: {:?}", histogram);

            let mut table = String::new();
            table.push_str("\\begin{table}[H]\n");
            table.push_str("\\centering\n");
            table.push_str("\\begin{tabular}{|c|c|}\n");
            table.push_str("    \\hline\n");
            table.push_str("    Reuse Interval & Portion \\\\ \n");
            table.push_str("    \\hline\n");
            for (k, v) in histogram {
                table.push_str(&format!("    {} & {:.3e} \\\\ \n", k, v));
            }
            table.push_str("    \\hline\n");
            table.push_str("\\end{tabular}\n");
            table.push_str("\\caption{Reuse Interval Distribution for ");
            table.push_str(name);
            table.push_str(&format!(" (block size {})", block_size));
            table.push_str("}\n");
            table.push_str("\\end{table}\n");
            println!("{}", table);
        });
}
