"""Study entrypoint. Defaults to the current (v2) external study.

  python -m strengthened.cli                 tests, train, analyse, paper, package
  python -m strengthened.cli --stage paper   regenerate the manuscript only
  python -m strengthened.cli --v1            the earlier eight-arm execution

v1 and v2 write to separate output directories and have separate fingerprints, so
running one never invalidates the other. Only v2 generates paper/emse/main.tex.
"""
import argparse
import subprocess
import sys
import unittest


def main():
    ap = argparse.ArgumentParser(description='Run the documented-data EMSE study')
    ap.add_argument('--stage', choices=['all', 'tests', 'train', 'analyze', 'paper', 'package'],
                    default='all')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--v1', action='store_true',
                    help='run the earlier eight-arm study instead of the current one')
    args = ap.parse_args()

    suffix = '' if args.v1 else '_v3'
    tests = f'strengthened.test_experiment{suffix}'
    module = f'strengthened.experiment{suffix}'

    if args.stage in ['all', 'tests']:
        suite = unittest.defaultTestLoader.loadTestsFromName(tests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)

    if args.stage in ['all', 'train']:
        cmd = [sys.executable, '-m', module, '--workers', str(args.workers)]
        if args.smoke:
            cmd.append('--smoke')
        subprocess.run(cmd, check=True)

    if args.smoke:
        if args.stage in ['analyze', 'paper', 'package']:
            raise SystemExit('Smoke results cannot generate a research manuscript')
        return

    if args.stage in ['all', 'analyze']:
        if args.v1:
            from .analyze import analyze
        else:
            from .analyze_v3 import analyze
        analyze()

    if args.stage in ['all', 'paper']:
        if args.v1:
            raise SystemExit('The v1 manuscript builder is superseded; it is kept for reference '
                             'only and no longer writes paper/emse. Drop --v1 to build the '
                             'current manuscript.')
        from .build_paper_v3 import build
        build()

    if args.stage in ['all', 'package']:
        if args.v1:
            raise SystemExit('Packaging applies to the current study only.')
        from .package_v3 import package
        package()


if __name__ == '__main__':
    main()
