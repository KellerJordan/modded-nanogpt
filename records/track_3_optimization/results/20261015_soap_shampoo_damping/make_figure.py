"""Generate a hardware-neutral SVG comparison directly from the submitted logs."""
from pathlib import Path
import statistics

from claim_stats import load_runs

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1040, 650
LEFT, RIGHT, TOP, BOTTOM = 100, 1000, 70, 560
XMIN, XMAX = 2600, 2700
YMIN, YMAX = 3.2735, 3.2855


def x(step):
    return LEFT + (step - XMIN) / (XMAX - XMIN) * (RIGHT - LEFT)


def y(loss):
    return BOTTOM - (loss - YMIN) / (YMAX - YMIN) * (BOTTOM - TOP)


def main():
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="16" fill="#222">',
             '<text x="550" y="32" text-anchor="middle" font-size="22">Track 3: SOAP-Shampoo with damping vs. #46</text>']
    for step in range(2600, 2701, 20):
        px = x(step)
        parts += [f'<path d="M {px} {TOP} V {BOTTOM}" stroke="#ddd"/>',
                  f'<text x="{px}" y="590" text-anchor="middle">{step}</text>']
    for loss in (3.274, 3.276, 3.278, 3.280, 3.282, 3.284):
        py = y(loss)
        parts += [f'<path d="M {LEFT} {py} H {RIGHT}" stroke="#ddd"/>',
                  f'<text x="85" y="{py + 6}" text-anchor="end">{loss:.3f}</text>']
    parts += [f'<rect x="{LEFT}" y="{TOP}" width="{RIGHT-LEFT}" height="{BOTTOM-TOP}" fill="none" stroke="#222"/>',
              '<text x="550" y="630" text-anchor="middle">step</text>',
              '<text transform="translate(28 315) rotate(-90)" text-anchor="middle">mean validation loss (Tail-EMA readout)</text>',
              f'<path d="M {LEFT} {y(3.28)} H {RIGHT}" stroke="black" stroke-width="1.5"/>']
    groups = [('SOAP-Shampoo + damping, seeds 49–64 (n=16)', '#1675b8', ROOT, range(49,65)),
              ('#46 comparison controls, seeds 17–24 (n=8)', '#d62728', ROOT/'controls_46', range(17,25))]
    for index, (label, color, folder, seeds) in enumerate(groups):
        runs = load_runs(folder, seeds)
        common = sorted(set.intersection(*(set(rows) for rows in runs.values())))
        points = [(step, statistics.mean(rows[step][1] for rows in runs.values()))
                  for step in common if XMIN <= step <= XMAX]
        coords = ' '.join(f'{x(step):.2f},{y(loss):.2f}' for step, loss in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        threshold = 3.28 - 0.004 / len(runs)**0.5
        parts.append(f'<path d="M {LEFT} {y(threshold)} H {RIGHT}" stroke="{color}" stroke-dasharray="2 3"/>')
        ly = 95 + index * 28
        parts += [f'<path d="M 400 {ly} h 35" stroke="{color}" stroke-width="3"/>',
                  f'<text x="445" y="{ly+5}">{label}</text>']
    parts += ['<text x="445" y="156" font-size="14">Dotted lines: significance thresholds for each batch</text>',
              '</g></svg>']
    (ROOT/'figure.svg').write_text('\n'.join(parts) + '\n')


if __name__ == '__main__':
    main()
