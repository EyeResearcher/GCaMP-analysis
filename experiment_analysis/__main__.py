import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare completed GCaMP recording bundles')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    from .workflow import run_experiment
    try:
        output = run_experiment(args.config)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Experiment analysis failed: {exc}\n')
    print(f'Experiment results: {output}')


if __name__ == '__main__':
    main()
