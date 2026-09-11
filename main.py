"""Notice for the removed combined workflow."""

def main(*args, **kwargs):
    raise SystemExit(
        "Combined execution has been removed. Run these stages separately:\n"
        "  python -m gcamp_analysis analyze RECORDINGS_ROOT --config ANALYSIS_CONFIG\n"
        "  python -m experiment_analysis run --config EXPERIMENT_CONFIG"
    )

if __name__ == "__main__":
    main()
