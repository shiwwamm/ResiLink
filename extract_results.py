#!/usr/bin/env python3
import json
from pathlib import Path

def extract(topology_id):
    summary_file = Path('optimization_summary.json')
    comparison_file = Path('network_evolution_comparison.json')
    
    if not summary_file.exists():
        print("No results found. Run test first.")
        return
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    with open(comparison_file) as f:
        comp = json.load(f)
    
    print(f"RESILINK RESULTS: {topology_id.upper()}")
    print("="*50)
    print(f"Initial Quality:      {summary['initial_quality']:.4f}")
    print(f"Final Quality:        {summary['final_quality']:.4f}")
    print(f"Improvement:          +{summary['quality_improvement']:.1%}")
    print(f"Links Suggested:      {summary['total_links_suggested']}")
    print(f"Algebraic Conn. Gain: {comp.get('algebraic_connectivity', {}).get('improvement', 0):+.1%}")
    print(f"Throughput Gain:      {comp.get('throughput_under_failure', {}).get('gain', 0):+.1%}")
    print(f"Latency Reduction:    {comp.get('avg_latency', {}).get('reduction', 0):+.1%}")

if __name__ == "__main__":
    import sys
    extract(sys.argv[1] if len(sys.argv) > 1 else "unknown")