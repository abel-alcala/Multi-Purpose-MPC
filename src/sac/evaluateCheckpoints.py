import glob
import json
import os
import re
import sys

_sacDir = os.path.dirname(os.path.abspath(__file__))
_srcDir = os.path.join(_sacDir, '..')
sys.path.insert(0, _srcDir)
sys.path.insert(0, _sacDir)
os.chdir(_srcDir)

from RL_Env import TrackingEnv
from agent import SACAgent, SACConfig
from action import ActionProcessor
from state import StateProcessor
from train import evaluate

checkpointDir = os.path.join(_sacDir, 'checkpoints')
outputPath = os.path.join(_sacDir, 'checkpoint_metrics.json')
evalEpisodes = 10


# Finds every saved checkpoint and pairs it with the step it came from, sorted by step
def findCheckpoints():
    found = []
    for path in glob.glob(os.path.join(checkpointDir, 'sac_*.pt')):
        match = re.match(r'sac_(\d+)\.pt$', os.path.basename(path))
        if match:
            found.append((int(match.group(1)), os.path.basename(path)))
    found.sort()

    # sac_final.pt has no step in its name, treat it as the last point
    if os.path.exists(os.path.join(checkpointDir, 'sac_final.pt')):
        lastStep = found[-1][0] if found else 0
        found.append((lastStep, 'sac_final.pt'))

    return found


# Evaluates every saved checkpoint and writes the metrics to a json file
def main(useObstacles=True):
    env = TrackingEnv(simMode='Sim_Track', useObstacles=useObstacles)
    stateProcessor = StateProcessor()
    actionProcessor = ActionProcessor()
    config = SACConfig(state_dim=7, action_dim=2, device='cpu')

    results = []

    # measure the untrained agent first so the curves have a real step 0 baseline
    # (typically ~100% collisions, 0% laps) instead of a made-up origin point
    baseline = evaluate(env, SACAgent(config), stateProcessor, actionProcessor, evalEpisodes)
    results.append({'step': 0, 'checkpoint': 'untrained', **baseline})
    print(f"untrained | return {baseline['mean_return']:.1f} | "
          f"laps {baseline['lap_rate']:.0%} | collisions {baseline['collision_rate']:.0%}")

    for step, filename in findCheckpoints():
        path = os.path.join(checkpointDir, filename)
        agent = SACAgent(config)
        agent.load(path)

        metrics = evaluate(env, agent, stateProcessor, actionProcessor, evalEpisodes)
        results.append({'step': step, 'checkpoint': filename, **metrics})

        print(f"{filename} | return {metrics['mean_return']:.1f} | "
              f"laps {metrics['lap_rate']:.0%} | collisions {metrics['collision_rate']:.0%}")

    with open(outputPath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"metrics saved to {outputPath}")
    env.close()


if __name__ == '__main__':
    main()
