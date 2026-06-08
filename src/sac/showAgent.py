import argparse
import os
import sys

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

_sacDir = os.path.dirname(os.path.abspath(__file__))
_srcDir = os.path.join(_sacDir, '..')
sys.path.insert(0, _srcDir)
sys.path.insert(0, _sacDir)
os.chdir(_srcDir)

from RL_Env import TrackingEnv
from agent import SACAgent, SACConfig
from action import ActionProcessor
from state import StateProcessor

checkpointDir = os.path.join(_sacDir, 'checkpoints')


# Loads a checkpoint and runs one rendered episode on the track
def runEpisode(checkpointFile, pause=0.03, useObstacles=True):
    path = os.path.join(checkpointDir, checkpointFile)
    if not os.path.exists(path):
        print(f"checkpoint not found: {path}")
        sys.exit(1)

    agent = SACAgent(SACConfig(state_dim=5, action_dim=2, device='cpu'))
    agent.load(path)

    stateProcessor = StateProcessor()
    actionProcessor = ActionProcessor()
    env = TrackingEnv(simMode='Sim_Track', renderMode='human', useObstacles=useObstacles)

    # slow down or speed up the render by overriding the pause length
    origPause = plt.pause
    plt.pause = lambda _: origPause(pause)

    obs, _ = env.reset()
    totalReturn = 0.0
    steps = 0
    done = False

    while not done:
        state = stateProcessor.normalize(obs)
        action = actionProcessor.scale_from_normalized(agent.select_action(state, deterministic=True))
        obs, reward, terminated, truncated, info = env.step(action)
        totalReturn += reward
        steps += 1
        done = terminated or truncated

    outcome = 'lap' if info['lap_complete'] else ('collision' if info['collision'] else 'timeout')
    print(f"steps {steps} | return {totalReturn:.1f} | {outcome}")

    plt.pause = origPause
    plt.show()
    env.close()


# Usage:
# python showAgent.py                                           shows agent running the saved sac_final.pt checkpoint
# python showAgent.py --checkpoint sac_150000.pt                runs a specific checkpoint
# python showAgent.py --slow                                    slower agent playback
# python showAgent.py --compare sac_50000.pt sac_final.pt       compare two checkpoints
# python showAgent.py --no-obstacles                            render without obstacles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='sac_final.pt')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    parser.add_argument('--no-obstacles', action='store_true')
    args = parser.parse_args()

    pause = 0.1 if args.slow else 0.03
    useObstacles = not args.no_obstacles

    if args.compare:
        runEpisode(args.compare[0], pause, useObstacles)
        runEpisode(args.compare[1], pause, useObstacles)
    else:
        runEpisode(args.checkpoint, pause, useObstacles)


if __name__ == '__main__':
    main()
