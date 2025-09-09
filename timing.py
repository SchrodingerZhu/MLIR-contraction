#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from collections import defaultdict, OrderedDict
import numpy as np

# Matplotlib is only needed for --mode plot
import matplotlib.pyplot as plt

DATA = {
    "Simulation Time": {
        "Tiled": {
            "3D Tensor Vector": "0:07.25",
            "4D Tensor": "2:52.02",
            "Attention Score": "29:31.60",
            "Batched GEMM": "2:50.64",
            "Context Lookup": "30:53.01",
            "Matrix Matrix": "0:40.00",
            "Matrix Vector": "0:11.10",
            "Rowwise SoftMax": "4:41.11"
        },
        "Original": {
            "3D Tensor Vector": "0:07.39",
            "4D Tensor": "12:27.31",
            "Attention Score": "59:35.35",
            "Batched GEMM": "8:14.36",
            "Context Lookup": "130:33.00",
            "Matrix Matrix": "3:34.90",
            "Matrix Vector": "0:11.32",
            "Rowwise SoftMax": "4:38.35"
        }
    },
    "Simulation Time (8-way assoc)": {
        "Tiled": {
            "3D Tensor Vector": "0:00.41",
            "4D Tensor": "0:00.72",
            "Attention Score": "0:07.57",
            "Batched GEMM": "0:01.33",
            "Context Lookup": "0:10.06",
            "Matrix Matrix": "0:00.68",
            "Matrix Vector": "0:00.43",
            "Rowwise SoftMax": "0:00.71"
        },
        "Original": {
            "3D Tensor Vector": "0:00.42",
            "4D Tensor": "0:00.73",
            "Attention Score": "0:07.35",
            "Batched GEMM": "0:01.42",
            "Context Lookup": "0:11.67",
            "Matrix Matrix": "0:00.71",
            "Matrix Vector": "0:00.43",
            "Rowwise SoftMax": "0:00.71"
        }
    },
    "SALT Time": {
        "Tiled": {
        "3D Tensor Vector": "0:00.005257",
        "4D Tensor": "0:00.005670",
        "Attention Score": "0:00.005679",
        "Batched GEMM": "0:00.006115",
        "Context Lookup": "0:00.005433",
        "Matrix Matrix": "0:00.005568",
        "Matrix Vector": "0:00.005698",
        "Rowwise SoftMax": "0:00.005639"
        },
        "Original": {
        "3D Tensor Vector": "0:00.005752",
        "4D Tensor": "0:00.005629",
        "Attention Score": "0:00.005222",
        "Batched GEMM": "0:00.006149",
        "Context Lookup": "0:00.005837",
        "Matrix Matrix": "0:00.005712",
        "Matrix Vector": "0:00.005319",
        "Rowwise SoftMax": "0:00.005307"
        },
        "Symbolic": {
        "3D Tensor Vector": "0:00.005568",
        "4D Tensor": "0:00.005389",
        "Attention Score": "0:00.006087",
        "Batched GEMM": "0:00.005905",
        "Context Lookup": "0:00.005440",
        "Matrix Matrix": "0:00.005592",
        "Matrix Vector": "0:00.005267",
        "Rowwise SoftMax": "0:00.005422"
        }
    },
    "Barvinok Time": {
        "Tiled": {
            "3D Tensor Vector": "00:01.46",
            "4D Tensor": "00:28.63",
            "Attention Score": "00:26.46",
            "Batched GEMM": "00:08.74",
            "Context Lookup": "00:30.02",
            "Matrix Matrix": "00:01.13",
            "Matrix Vector": "00:00.22",
            "Rowwise SoftMax": "00:02.80"
        },
        "Original": {
            "3D Tensor Vector": "00:00.53",
            "4D Tensor": "00:01.49",
            "Attention Score": "00:03.03",
            "Batched GEMM": "00:01.31",
            "Context Lookup": "00:03.61",
            "Matrix Matrix": "00:00.58",
            "Matrix Vector": "00:00.21",
            "Rowwise SoftMax": "00:00.79"
        },
        "Symbolic": {
            "3D Tensor Vector": "00:02.02",
            "4D Tensor": "00:06.02",
            "Attention Score": "00:44.01",
            "Batched GEMM": "00:06.02",
            "Context Lookup": "00:37.45",
            "Matrix Matrix": "00:00.59",
            "Matrix Vector": "00:00.46",
            "Rowwise SoftMax": "00:08.25"
        }
    }
}

KERNEL_ORDER = [
    "3D Tensor Vector",
    "4D Tensor", 
    "Attention Score",
    "Batched GEMM",
    "Context Lookup",
    "Matrix Matrix",
    "Matrix Vector",
    "Rowwise SoftMax",
]

def parse_time_to_seconds(s: str) -> float:
    """
    Accepts 'm:ss', 'mm:ss', 'mm:ss.xx', or 'h:mm:ss(.xx)' just in case.
    Returns seconds (float).
    """
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    elif len(parts) == 3:  # support h:mm:ss(.xx)
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    else:
        raise ValueError(f"Unrecognized time format: {s}")

def tidy_rows(data):
    """
    Convert nested dict to a tidy list of rows:
      (family, variant, kernel, seconds, raw_str)
    """
    rows = []
    for family, variants in data.items():
        for variant, kernel_map in variants.items():
            for kernel, raw in kernel_map.items():
                rows.append((family, variant, kernel, parse_time_to_seconds(raw), raw))
    return rows

def families_variants(data):
    """Ordered mapping: family -> list of variants (preserve input order)."""
    fam2vars = OrderedDict()
    for family, variants in data.items():
        fam2vars[family] = list(variants.keys())
    return fam2vars

def plot_all(data, out_path: str, dpi=200, style="default"):
    # Set the plot style
    if style != "default":
        try:
            plt.style.use(style)
        except OSError:
            print(f"Warning: Style '{style}' not found. Using default style.")
    
    rows = tidy_rows(data)
    fam2vars = families_variants(data)

    # Prepare data: family -> variant -> list aligned to KERNEL_ORDER
    fam_variant_kernel_seconds = {}
    fam_variant_kernel_raw = {}
    for family in fam2vars:
        fam_variant_kernel_seconds[family] = {}
        fam_variant_kernel_raw[family] = {}
        for variant in fam2vars[family]:
            vals_sec = []
            vals_raw = []
            for k in KERNEL_ORDER:
                raw = data[family][variant][k]
                vals_raw.append(raw)
                vals_sec.append(parse_time_to_seconds(raw))
            fam_variant_kernel_seconds[family][variant] = vals_sec
            fam_variant_kernel_raw[family][variant] = vals_raw

    n_families = len(fam2vars)
    fig_height = 3.0 * n_families + 1.0  # Increased height for better spacing
    fig, axes = plt.subplots(n_families, 1, figsize=(14, fig_height), sharex=True)
    if n_families == 1:
        axes = [axes]

    # Horizontal bar charts with grouped offsets per variant
    for ax, (family, variants) in zip(axes, fam2vars.items()):
        base_y = range(len(KERNEL_ORDER))
        nvar = len(variants)
        # Increased offset for better spacing
        offset = 0.3
        start = -offset * (nvar - 1) / 2.0

        for i, var in enumerate(variants):
            y = [y0 + start + i * offset for y0 in base_y]
            x = fam_variant_kernel_seconds[family][var]
            bars = ax.barh(y, x, height=0.18, label=var)
            
            # Add value labels at the end of each bar
            for j, (bar, value) in enumerate(zip(bars, x)):
                # Format value in scientific notation
                if value >= 1:
                    # For values >= 1, use standard scientific notation
                    formatted_value = f"{value:.2e}"
                else:
                    # For values < 1, use more readable format
                    formatted_value = f"{value:.2e}"
                
                # Adjust text position based on bar position
                text_x = value * 1.1  # Position text slightly to the right of bar end
                
                # Vertical offset based on position in the group
                if i == 0:  # First variant (top bars)
                    text_y = y[j] - 0.0618  # Lift text up
                elif i == nvar - 1:  # Last variant (bottom bars)
                    text_y = y[j] + 0.0618  # Push text down
                else:  # Middle variants
                    text_y = y[j]  # Keep text centered
                
                ax.text(text_x, text_y, formatted_value, 
                       ha='left', va='center', fontsize=8)

        ax.set_yticks(list(base_y))
        ax.set_yticklabels(KERNEL_ORDER)
        ax.invert_yaxis()  # Invert y-axis so first kernel appears at top
        ax.set_title(family)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        ax.legend(frameon=False, ncols=min(3, nvar))
        ax.set_xscale('log')

    axes[-1].set_xlabel("Time (seconds, log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

def latex_table(data):
    """
    Emit a LaTeX table (booktabs) with two header rows:
    Families as top-level columns; their variants as subcolumns.
    Values are the original mm:ss.xx strings.
    """
    fam2vars = families_variants(data)

    # Column spec: first column is kernel, then one col per variant of each family
    cols = ["l"]
    for family, variants in fam2vars.items():
        cols.extend(["c"] * len(variants))
    colspec = "".join(cols)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")

    # First header row: families with \multicolumn
    header_top = [r"\textbf{Kernel}"]
    for family, variants in fam2vars.items():
        header_top.append(r"\multicolumn{" + str(len(variants)) + r"}{c}{\textbf{" + family + r"}}")
    lines.append(" & ".join(header_top) + r" \\")
    lines.append(r"\midrule")

    # Second header row: variants
    header_mid = [""]
    for family, variants in fam2vars.items():
        for v in variants:
            header_mid.append(r"\textit{" + v + r"}")
    lines.append(" & ".join(header_mid) + r" \\")
    lines.append(r"\midrule")

    # Body
    for kernel in KERNEL_ORDER:
        row = [kernel]
        for family, variants in fam2vars.items():
            for v in variants:
                row.append(data[family][v][kernel])
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Timing comparison across kernels and methods.}")
    lines.append(r"\label{tab:kernel-times}")
    lines.append(r"\end{table}")

    print("\n".join(lines))

def speedup_table(data):
    """
    Compute geometric mean speedup of SALT over other methods and emit LaTeX table.
    """
    # Find SALT Time family
    salt_family = None
    for family in data.keys():
        if "SALT" in family.upper():
            salt_family = family
            break
    
    if not salt_family:
        print("Error: No SALT Time family found in data.")
        return
    
    # Get SALT times for all kernels and variants
    salt_data = data[salt_family]
    
    # Prepare speedup calculations
    speedup_results = {}
    
    # For each other family, calculate speedup vs SALT
    for family, variants in data.items():
        if family == salt_family:
            continue
            
        speedup_results[family] = {}
        
        for variant in variants.keys():
            if variant not in salt_data:
                continue
                
            speedups = []
            
            for kernel in KERNEL_ORDER:
                if kernel in variants[variant] and kernel in salt_data[variant]:
                    other_time = parse_time_to_seconds(variants[variant][kernel])
                    salt_time = parse_time_to_seconds(salt_data[variant][kernel])
                    
                    if salt_time > 0:  # Avoid division by zero
                        speedup = other_time / salt_time
                        speedups.append(speedup)
            
            if speedups:
                # Calculate geometric mean
                geomean = np.exp(np.mean(np.log(speedups)))
                speedup_results[family][variant] = geomean
    
    # Generate LaTeX table (rotated: variants as rows, families as columns)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    
    # Get all unique variants across all families
    all_variants = set()
    for family, variants in speedup_results.items():
        all_variants.update(variants.keys())
    all_variants = sorted(all_variants)
    
    # Column spec: Variant + one column per family
    num_families = len(speedup_results)
    colspec = "l" + "c" * num_families
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    
    # Header row: Variant + family names
    header = [r"\textbf{Variant}"]
    for family in sorted(speedup_results.keys()):
        header.append(r"\textbf{" + family + r"}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    
    # Body: one row per variant
    for variant in all_variants:
        row = [r"\textit{" + variant + r"}"]
        for family in sorted(speedup_results.keys()):
            if variant in speedup_results[family]:
                speedup = speedup_results[family][variant]
                row.append(f"{speedup:.2f}x")
            else:
                row.append("--")  # No data available
        lines.append(" & ".join(row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}%")
    lines.append(r"\caption{Geometric mean speedup of other methods over SALT.}")
    lines.append(r"\label{tab:salt-speedup}")
    lines.append(r"\end{table}")
    
    print("\n".join(lines))

def main():
    p = argparse.ArgumentParser(description="Plot or export LaTeX table for kernel timing data.")
    p.add_argument("--mode", choices=["plot", "latex", "speedup"], required=True,
                   help="plot: save a multi-panel bar chart; latex: print LaTeX table to stdout; speedup: print geometric mean speedup table of other methods over SALT.")
    p.add_argument("--out", default="times.png",
                   help="Output path for --mode plot (ignored for --mode latex).")
    p.add_argument("--style", default="default",
                   help="Plot style to use. Options: default, ggplot, seaborn, fivethirtyeight, bmh, dark_background, etc. Use --list-styles to see all available styles (ignored for --mode latex).")
    p.add_argument("--list-styles", action="store_true",
                   help="List all available matplotlib styles and exit.")
    args = p.parse_args()

    if args.list_styles:
        print("Available matplotlib styles:")
        print("=" * 40)
        styles = plt.style.available
        for i, style in enumerate(sorted(styles), 1):
            print(f"{i:2d}. {style}")
        print(f"\nTotal: {len(styles)} styles available")
        print("\nUsage examples:")
        print("  python timing.py --mode plot --style ggplot")
        print("  python timing.py --mode plot --style seaborn")
        print("  python timing.py --mode plot --style fivethirtyeight")
        return

    if args.mode == "plot":
        plot_all(DATA, args.out, style=args.style)
    elif args.mode == "latex":
        latex_table(DATA)
    elif args.mode == "speedup":
        speedup_table(DATA)

if __name__ == "__main__":
    main()
