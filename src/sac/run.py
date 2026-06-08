import argparse
import os
import sys

_sacDir = os.path.dirname(os.path.abspath(__file__))
_srcDir = os.path.join(_sacDir, '..')
sys.path.insert(0, _srcDir)
sys.path.insert(0, _sacDir)
os.chdir(_srcDir)

import train
import evaluateCheckpoints
import plotTraining
import showAgent

# Usage:
#   python run.py                     train, evaluate, plot, then visualize
#   python run.py --no-obstacles      same as above with no obstacles on the track : agent can only make it all the way through with no obstacles rn :(
#   python run.py --viz-only          skip training and just visualize saved checkpoints (to not have to rerun training each time)
#   python run.py --slow --viz-only   agent visualization runs slower using saved checkpoints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-obstacles', action='store_true', help='run without obstacles on the track')
    parser.add_argument('--viz-only', action='store_true', help='skip training and only visualize')
    parser.add_argument('--checkpoint', default='sac_final.pt', help='checkpoint to visualize')
    parser.add_argument('--slow', action='store_true', help='slower playback during visualization')
    args = parser.parse_args()

    useObstacles = not args.no_obstacles

    # train, evaluate every checkpoint, then plot the curves
    if not args.viz_only:
        train.train(useObstacles)
        evaluateCheckpoints.main(useObstacles)
        plotTraining.main()

    # watch the trained agent drive the track
    pause = 0.1 if args.slow else 0.03
    showAgent.runEpisode(args.checkpoint, pause, useObstacles)


if __name__ == '__main__':
    main()