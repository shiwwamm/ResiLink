# results_table.py
import json
from pathlib import Path

report = json.load(open('real_world_test_report_research.json'))

print("Topology | Nodes | Links Added | Resilience Gain | Meets Target")
print("-"*70)

for r in report['test_results']:
    info = r['topology_info']
    opt = r['optimization_result']
    if not opt['success']: continue
    
    summary = opt.get('summary', {})
    gain = summary.get('quality_improvement', 0)
    target = info['expected_improvement']
    meets = "Yes" if gain >= 0.7 * target else "No"
    
    print(f"{info['name'][:15]:15} | {info['nodes']:5} | {summary.get('total_links_suggested',0):11} | {gain:+6.1%} | {meets}")