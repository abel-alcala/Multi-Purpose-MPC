import json
import os

import matplotlib.pyplot as plt

_sacDir = os.path.dirname(os.path.abspath(__file__))
metricsPath = os.path.join(_sacDir, 'checkpoint_metrics.json')
outputPath = os.path.join(_sacDir, 'trainingPlots.png')


# Reads the checkpoint metrics and plots the four training curves
def main():
    if not os.path.exists(metricsPath):
        print(f"no metrics file at {metricsPath}, run evaluateCheckpoints.py first")
        return

    with open(metricsPath) as f:
        records = json.load(f)

    # sac_final.pt and sac_300000.pt land on the same step, drop the duplicate
    seen = set()
    unique = []
    for r in records:
        key = (r['step'], r['checkpoint'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # step 0 is the real untrained baseline measured in evaluateCheckpoints, so the
    # curves are anchored at genuine values rather than a made-up origin point
    steps = [r['step'] for r in unique]
    meanReturn = [r['mean_return'] for r in unique]
    meanAbsEy = [r['mean_abs_ey'] for r in unique]
    lapRate = [r['lap_rate'] * 100 for r in unique]
    collisionRate = [r['collision_rate'] * 100 for r in unique]

    # format the x axis ticks as e.g. 50k, 100k
    fmt = lambda x, _: f'{int(x / 1000)}k'

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SAC Training Progress', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(steps, meanReturn, 'o-', color='#2563eb', linewidth=2, markersize=6)
    ax.set_title('Mean Episode Return')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Return')
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))

    ax = axes[0, 1]
    ax.plot(steps, meanAbsEy, 'o-', color='#dc2626', linewidth=2, markersize=6)
    ax.set_title('Mean Absolute Lateral Error')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('|e_y| (m)')
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))

    ax = axes[1, 0]
    ax.plot(steps, lapRate, 'o-', color='#16a34a', linewidth=2, markersize=6)
    ax.set_title('Lap Completion Rate')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Lap Rate (%)')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))

    ax = axes[1, 1]
    ax.plot(steps, collisionRate, 'o-', color='#d97706', linewidth=2, markersize=6)
    ax.set_title('Collision Rate')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Collision Rate (%)')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))

    plt.tight_layout()
    plt.savefig(outputPath, dpi=150, bbox_inches='tight')
    print(f"plot saved to {outputPath}")
    plt.show()


if __name__ == '__main__':
    main()
