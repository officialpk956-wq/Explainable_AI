"""Run the current EMSE study; --earlier selects the prior corrected experiment."""
import sys

if __name__ == '__main__':
    if '--earlier' in sys.argv:
        sys.argv.remove('--earlier')
        from revised.cli import main
    else:
        from strengthened.cli import main
    raise SystemExit(main())
