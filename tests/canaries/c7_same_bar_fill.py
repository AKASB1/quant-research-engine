"""C7 same_bar_fill: an ordinary built-in strategy (signal_x_ls) run with unsafe_same_bar_fill,
so its orders fill at the close of the bar in which the decision was taken. Guard: G5 (the run
is labelled UNSAFE in every output and the report generator refuses it)."""

CANARY = {
    "id": "C7",
    "name": "same_bar_fill",
    "guard": "G5",
    "module": "quant_research_engine.strategies.signal_x_ls",
    "class": "SignalXLS",
    "kind": "run",
    "run": {"fill_model": "same_close", "unsafe_same_bar_fill": True},
}
