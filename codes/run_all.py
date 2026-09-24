#!/usr/bin/env python3
"""Regenere toutes les figures numeriques du document, dans ../figs/."""
import runpy, pathlib, sys, time

SCRIPTS = [
    "scalar_quantized_demo.py",
    "quantized_state_feedback.py",
    "make_tradeoff_fig.py",
    "transport_backstepping_demo.py",
    "transport_quantized_control.py",
    "delay_quantized_predictor.py",
]
here = pathlib.Path(__file__).resolve().parent
for s in SCRIPTS:
    print(f"=== {s} ===", flush=True)
    t0 = time.time()
    runpy.run_path(str(here / s), run_name="__main__")
    print(f"    termine en {time.time()-t0:.1f}s", flush=True)
print("Toutes les figures sont dans", (here.parent / "figs").resolve())
