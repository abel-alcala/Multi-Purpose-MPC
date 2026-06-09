import glob
import os
import sys
from collections import deque

import numpy as np

_srcDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _srcDir)
# RL_Env uses relative paths for map files, so the working directory must be src/
os.chdir(_srcDir)

from RL_Env import TrackingEnv
from agent import SACAgent, SACConfig
from replay_buffer import ReplayBuffer
from action import ActionProcessor
from state import StateProcessor

# training hyperparameters
totalSteps = 900000
warmupSteps = 5000  # random actions before the first gradient update
batchSize = 256
evalFreq = 10000  # run an evaluation every N steps
evalEpisodes = 10  # episodes per evaluation
checkpointFreq = 20000  # save weights every N steps once past the early phase
earlyCheckpointFreq = 5000  # save more often early on, where the policy changes fast
earlyPhase = 50000  # how long to keep the denser early checkpoint rate
logFreq = 1000  # print rolling stats every N steps
bufferCapacity = 1000000
earlyStopEvals = 20  # stop after this many consecutive perfect evals

checkpointDir = os.path.join(os.path.dirname(__file__), 'checkpoints')


# Runs n deterministic episodes and returns the averaged metrics
def evaluate(env, agent, stateProcessor, actionProcessor, nEpisodes):
    returns = []
    lateralErrors = []
    laps = 0
    collisions = 0

    for _ in range(nEpisodes):
        obs, _ = env.reset()
        epReturn = 0.0
        done = False

        while not done:
            state = stateProcessor.normalize(obs)
            normAction = agent.select_action(state, deterministic=True)
            rawAction = actionProcessor.scale_from_normalized(normAction)

            obs, reward, terminated, truncated, info = env.step(rawAction)
            epReturn += reward
            lateralErrors.append(abs(info['e_y']))
            done = terminated or truncated

        returns.append(epReturn)
        if info['lap_complete']:
            laps += 1
        if info['collision']:
            collisions += 1

    return {
        'mean_return': float(np.mean(returns)),
        'mean_abs_ey': float(np.mean(lateralErrors)),
        'lap_rate': laps / nEpisodes,
        'collision_rate': collisions / nEpisodes,
    }


# Trains the SAC agent on the tracking environment and saves checkpoints along the way
def train(useObstacles=True, initCheckpoint=None):
    os.makedirs(checkpointDir, exist_ok=True)

    stateProcessor = StateProcessor()
    actionProcessor = ActionProcessor()

    config = SACConfig(state_dim=stateProcessor.state_dim, action_dim=actionProcessor.action_dim, device='cpu')
    agent = SACAgent(config)
    buffer = ReplayBuffer(state_dim=stateProcessor.state_dim, action_dim=actionProcessor.action_dim, capacity=bufferCapacity)

    # Load starting checkpoint before deleting old files so it isn't wiped
    if initCheckpoint is not None:
        agent.load(initCheckpoint)
        print(f"started from {initCheckpoint}")

    # Clear numbered checkpoints from a previous run so the plot only covers the current run
    for old in glob.glob(os.path.join(checkpointDir, 'sac_[0-9]*.pt')):
        os.remove(old)

    env = TrackingEnv(simMode='Sim_Track', useObstacles=useObstacles)
    # A separate environment is used for evaluation so evaluation episodes never affect the training state
    evalEnv = TrackingEnv(simMode='Sim_Track', useObstacles=useObstacles)

    obs, _ = env.reset()
    normState = stateProcessor.normalize(obs)

    epReturn = 0.0
    epSteps = 0
    epCount = 0
    recentReturns = deque(maxlen=20)
    perfectEvals = 0  # consecutive evals with every lap clean

    print(f"obstacles {'on' if useObstacles else 'off'}, training for {totalSteps} steps")

    for step in range(1, totalSteps + 1):

        # pick a random action during warmup, otherwise let the agent decide
        if step <= warmupSteps:
            rawAction = env.action_space.sample()
            normAction = actionProcessor.normalize_raw(rawAction)
        else:
            normAction = agent.select_action(normState, deterministic=False)
            rawAction = actionProcessor.scale_from_normalized(normAction)

        nextObs, reward, terminated, truncated, info = env.step(rawAction)
        normNextState = stateProcessor.normalize(nextObs)

        # store the normalized transition and move on
        buffer.add(normState, normAction, reward, normNextState, float(terminated))
        normState = normNextState
        epReturn += reward
        epSteps += 1

        # episode finished, log it and reset
        if terminated or truncated:
            epCount += 1
            recentReturns.append(epReturn)
            result = 'lap' if info['lap_complete'] else ('collision' if info['collision'] else 'timeout')
            print(f"ep {epCount} | steps {epSteps} | return {epReturn:.1f} | {result}")

            obs, _ = env.reset()
            normState = stateProcessor.normalize(obs)
            epReturn = 0.0
            epSteps = 0

        # one gradient update per step once warmup is over and the buffer has enough
        if step > warmupSteps and len(buffer) >= batchSize:
            agent.update(buffer, batchSize)

        # rolling return log
        if step % logFreq == 0 and recentReturns:
            print(f"step {step} | mean return {np.mean(recentReturns):.1f}")

        # Evaluate periodically and stop early if the agent laps cleanly enough times in a row
        if step % evalFreq == 0:
            metrics = evaluate(evalEnv, agent, stateProcessor, actionProcessor, evalEpisodes)
            print(f"eval at step {step} | return {metrics['mean_return']:.1f} | "
                  f"laps {metrics['lap_rate']:.0%} | collisions {metrics['collision_rate']:.0%}")

            if metrics['lap_rate'] == 1.0 and metrics['collision_rate'] == 0.0:
                perfectEvals += 1
                if perfectEvals >= earlyStopEvals:
                    print(f"{earlyStopEvals} clean evals in a row, stopping early")
                    break
            else:
                perfectEvals = 0

        # save a checkpoint, densely during the early phase and at the regular rate after
        earlyCheckpoint = step <= earlyPhase and step % earlyCheckpointFreq == 0
        if earlyCheckpoint or step % checkpointFreq == 0:
            path = os.path.join(checkpointDir, f'sac_{step}.pt')
            agent.save(path)
            print(f"saved {path}")

    # save the final policy
    path = os.path.join(checkpointDir, 'sac_final.pt')
    agent.save(path)
    print(f"done, final checkpoint saved to {path}")
    env.close()
    evalEnv.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-obstacles', action='store_true', help='train without obstacles on the track')
    parser.add_argument('--init-checkpoint', default=None, help='start the agent from this checkpoint')
    args = parser.parse_args()
    train(useObstacles=not args.no_obstacles, initCheckpoint=args.init_checkpoint)
