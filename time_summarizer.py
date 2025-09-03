#!/usr/bin/env python3
"""
Time Summarizer Script
Analyzes all .json files in the workspace and summarizes timing information.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

def load_json_file(filepath: str) -> Dict:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return {}

def extract_timing_info(data: Dict) -> Tuple[float, bool]:
    """
    Extract timing information from JSON data.
    Returns (total_time_ms, has_timing_data)
    """
    if 'analysis_time' not in data:
        return 0.0, False
    
    timing = data['analysis_time']
    if 'secs' not in timing or 'nanos' not in timing:
        return 0.0, False
    
    # Convert to milliseconds
    total_ms = timing['secs'] * 1000 + timing['nanos'] / 1_000_000
    return total_ms, True

def get_relative_path(filepath: str, root_dir: str) -> str:
    """Get relative path from root directory."""
    return os.path.relpath(filepath, root_dir)

def categorize_file(relative_path: str) -> str:
    """Categorize file based on its path."""
    if relative_path.startswith('symbolic/'):
        return 'symbolic'
    elif relative_path.startswith('constant/tiled/'):
        return 'constant_tiled'
    elif relative_path.startswith('constant/'):
        return 'constant'
    else:
        return 'other'

def main():
    # Get the root directory (current working directory)
    root_dir = os.getcwd()
    print(f"Analyzing JSON files in: {root_dir}")
    print("=" * 80)
    
    # Find all JSON files
    json_files = glob.glob('**/*.json', recursive=True)
    
    if not json_files:
        print("No JSON files found!")
        return
    
    print(f"Found {len(json_files)} JSON files\n")
    
    # Data structures for analysis
    timing_data = []
    category_data = {
        'symbolic': [],
        'constant': [],
        'constant_tiled': [],
        'other': []
    }
    
    # Process each file
    for filepath in sorted(json_files):
        relative_path = get_relative_path(filepath, root_dir)
        data = load_json_file(filepath)
        
        if not data:
            continue
            
        timing_ms, has_timing = extract_timing_info(data)
        category = categorize_file(relative_path)
        
        if has_timing:
            timing_data.append({
                'file': relative_path,
                'time_ms': timing_ms,
                'category': category
            })
            category_data[category].append(timing_ms)
        
        # Print individual file info
        if has_timing:
            print(f"{relative_path:60} | {timing_ms:8.3f} ms | {category}")
        else:
            print(f"{relative_path:60} | No timing data | {category}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if not timing_data:
        print("No timing data found in any files!")
        return
    
    # Overall statistics
    all_times = [item['time_ms'] for item in timing_data]
    print(f"\nOverall Statistics ({len(all_times)} files with timing data):")
    print(f"  Total time:    {sum(all_times):8.3f} ms")
    print(f"  Average time:  {statistics.mean(all_times):8.3f} ms")
    print(f"  Median time:   {statistics.median(all_times):8.3f} ms")
    print(f"  Min time:      {min(all_times):8.3f} ms")
    print(f"  Max time:      {max(all_times):8.3f} ms")
    if len(all_times) > 1:
        print(f"  Std deviation: {statistics.stdev(all_times):8.3f} ms")
    
    # Category statistics
    print(f"\nBy Category:")
    for category, times in category_data.items():
        if times:
            print(f"\n  {category.upper()}:")
            print(f"    Count:       {len(times):8d}")
            print(f"    Total time:  {sum(times):8.3f} ms")
            print(f"    Average:     {statistics.mean(times):8.3f} ms")
            print(f"    Median:      {statistics.median(times):8.3f} ms")
            print(f"    Min:         {min(times):8.3f} ms")
            print(f"    Max:         {max(times):8.3f} ms")
            if len(times) > 1:
                print(f"    Std dev:     {statistics.stdev(times):8.3f} ms")
    
    # Top 10 slowest files
    print(f"\nTop 10 Slowest Files:")
    slowest_files = sorted(timing_data, key=lambda x: x['time_ms'], reverse=True)[:10]
    for i, item in enumerate(slowest_files, 1):
        print(f"  {i:2d}. {item['file']:50} | {item['time_ms']:8.3f} ms | {item['category']}")
    
    # Top 10 fastest files
    print(f"\nTop 10 Fastest Files:")
    fastest_files = sorted(timing_data, key=lambda x: x['time_ms'])[:10]
    for i, item in enumerate(fastest_files, 1):
        print(f"  {i:2d}. {item['file']:50} | {item['time_ms']:8.3f} ms | {item['category']}")

if __name__ == "__main__":
    main()
