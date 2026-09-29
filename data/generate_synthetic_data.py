"""Generate a synthetic SD-WAN hub dataset with the same JSON schema as the
original Cisco vManage interface-statistics export.

The real data is confidential; the values here are random and only mimic the
structure (daily aggregates per interface) so the notebook can be run end to end.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

days = pd.date_range("2024-03-01", periods=48, freq="D")
# (tx mean, rx mean) in kbps per interface — synthetic profiles
profiles = {
    "10ge0/0": (900, 1),       # WAN uplink, mostly TX
    "ge2/0": (950, 5000),      # LAN-facing, heavy RX
    "ge2/1": (1200, 5200),     # LAN-facing, heavy RX
    "mgmt0": (0, 11),          # management port
    "system": (0, 0),          # system interface (filtered out during cleaning)
}

rows = []
for i, day in enumerate(days):
    weekend = day.dayofweek >= 5
    trend = 1 + 0.006 * i                     # slow growth over the period
    load = (0.55 if weekend else 1.0) * trend
    count = int(rng.integers(270, 292)) if i not in (0, len(days) - 1) else int(rng.integers(195, 260))
    for iface, (tx, rx) in profiles.items():
        tx_v = max(0.0, tx * load * rng.normal(1, 0.12)) if tx else 0.0
        rx_v = max(0.0, rx * load * rng.normal(1, 0.15)) if rx > 20 else float(rx)
        if iface == "mgmt0":
            rx_v = 11 + rng.normal(0, 0.01)
        # a few spikes to give the anomaly detector something to find
        if iface.startswith("ge") and rng.random() < 0.05:
            rx_v *= rng.uniform(1.4, 1.8)
        rows.append({
            "entry_time": int(day.timestamp() * 1000),
            "count": count,
            "interface": iface,
            "tx_kbps": round(tx_v, 6),
            "rx_kbps": round(rx_v, 6),
        })

out = Path(__file__).parent / "sample_hub_data.json"
out.write_text(json.dumps({"data": rows}, indent=1))
print(f"Wrote {len(rows)} rows to {out}")
