import json
from src.api.solver_api import _load_task_grids
from src.reasoning.rule_discovery.rule_discoverer import RuleDiscoverer
from src.reasoning.planner.task_planner import TaskPlanner

# Test on solved tasks and a variety of others
test_ids = ['1cf80156', '1e0a9b12', '0d3d703e', '007bbfb7']

for tid in test_ids:
    try:
        task_dict = json.load(open(f'data/datasets/training/{tid}.json'))
        task = _load_task_grids(task_dict)
        pairs = task['train']
        disc = RuleDiscoverer()
        hyps = disc.discover(pairs)
        sizes = [(p["input"].size, p["output"].size) for p in pairs]
        print(f'\n{tid}: {len(hyps)} hypotheses, {len(pairs)} pairs')
        print(f'  Sizes: {sizes}')
        for h in hyps[:3]:
            print(f'  - {h}')
        if not hyps:
            print('  [No hypotheses found - task likely requires unsupported primitives]')
    except Exception as e:
        print(f'{tid}: ERROR {e}')
