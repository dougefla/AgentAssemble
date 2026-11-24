"""`python -m evo` → the interactive evo CLI (same as the `evo` console script)."""
from evo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
