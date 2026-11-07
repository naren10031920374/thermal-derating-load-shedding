#!/usr/bin/env python3
"""
Trace figures for the six new load-pattern scenarios, in the same style as the sensor-margin figures
(make_sensor_graphs.py): the 10 bus voltages (top) and the load asked for / load the limited bus got (bottom).
Every curve comes from the trace_<ID>_tr_p6_o<offset>_*.csv files written by run_staged_limiter.m with save_trace on
(see submit_trace_profile6.sh). Nothing is simulated or invented here.

Offset +5 = margin 0.  Each panel is one single simulation.

    python3 make_profile6_graphs.py --dir model_outputs/dataset_pipeline --out profile6_graphs
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_sensor_graphs as m

NOISE = "10% noise, 1 s average"

def panel(sid, buses, off, note):
    margin = {5: "margin 0"}.get(off, "margin %s kW" % ("−%d" % (off - 5)))
    word = "buses" if ";" in buses else "bus"
    return (sid, "tr_p6_o%d" % off, "%s (%s %s), %s\n%s, %s" % (sid, word, buses, margin, NOISE, note))

FIGS = [
    ("p6_RT051", [panel("RT051", "L", 5, "offset +5"), panel("RT051", "L", 7, "largest safe offset +7"),
                  panel("RT051", "L", 8, "1 kW higher, +8")]),
    ("p6_RC034", [panel("RC034", "F", 5, "offset +5"), panel("RC034", "F", 10, "offset +10"),
                  panel("RC034", "F", 15, "offset +15")]),
    ("p6_RT057", [panel("RT057", "K;L", 5, "offset +5 (largest safe)"), panel("RT057", "K;L", 6, "1 kW higher, +6"),
                  panel("RT057", "K;L", 10, "offset +10")]),
    ("p6_RC007", [panel("RC007", "E;F", 5, "offset +5"), panel("RC007", "E;F", 10, "offset +10")]),
    ("p6_RT071", [panel("RT071", "A;B;C", 5, "offset +5"), panel("RT071", "A;B;C", 9, "largest safe offset +9"),
                  panel("RT071", "A;B;C", 10, "1 kW higher, +10")]),
    ("p6_RT041", [panel("RT041", "D;E;F", 5, "offset +5"), panel("RT041", "D;E;F", 10, "largest safe offset +10"),
                  panel("RT041", "D;E;F", 11, "1 kW higher, +11")]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="model_outputs/dataset_pipeline")
    ap.add_argument("--out", default="profile6_graphs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, panels in FIGS:
        m.fig(a.dir, a.out, name, panels)
