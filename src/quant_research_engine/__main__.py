"""Entry point: ``python -m quant_research_engine`` (threads pinned before NumPy loads)."""

from quant_research_engine._threads import pin_threads

pin_threads()

if __name__ == "__main__":
    import sys

    from quant_research_engine.cli import main

    sys.exit(main())
