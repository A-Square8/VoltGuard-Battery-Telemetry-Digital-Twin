"""
VoltGuard Data Simulator — Dataset-Backed
Replays real NASA Battery Aging Dataset rows for normal operation.
Applies instant fault modifications for fault injection scenarios.
"""

import os
import numpy as np
import pandas as pd
import time
from collections import deque

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, 'discharge.csv')


class DatasetSimulator:
    """
    Replays real telemetry from discharge.csv.
    Fault injection modifies real readings to simulate dangerous conditions.
    """

    FAULT_DESCRIPTIONS = {
        "normal": {
            "title": "NOMINAL OPERATION",
            "detail": "All sensors within safe operating range. Battery performing normally.",
            "expected": "V: 3.5-4.0V | I: -2.0A | T: 24-34°C",
        },
        "overheat": {
            "title": "DANGER: BATTERY OVERHEATING",
            "detail": (
                "Battery temperature spiked to 75-85°C, far exceeding the safe limit of 40°C. "
                "This simulates a thermal runaway event — an uncontrolled exothermic reaction "
                "inside the cell. In a real system, this triggers emergency cooling and circuit disconnection."
            ),
            "expected": "Temperature: 75-85°C | Isolation Forest: THERMAL RUNAWAY | Health: DEGRADED",
        },
        "voltage_sag": {
            "title": "WARNING: DEEP DISCHARGE",
            "detail": (
                "Battery voltage has dropped to 0.5-1.5V, far below the minimum safe threshold of 2.5V. "
                "The cell is critically depleted. Continued operation at this voltage risks permanent "
                "damage to the anode structure and irreversible capacity loss."
            ),
            "expected": "Voltage: 0.5-1.5V | XGBoost: CRITICAL | Health Score: 20%",
        },
        "unstable": {
            "title": "WARNING: ERRATIC CURRENT DRAW",
            "detail": (
                "Current is oscillating wildly between -8A and +8A instead of the normal -2A discharge. "
                "This indicates a possible short circuit, faulty load controller, or failing BMS. "
                "High current fluctuations cause internal heating and accelerated cell degradation."
            ),
            "expected": "Current: -8 to +8A | Feature: current_instability spike | Isolation Forest: ANOMALY",
        },
    }

    def __init__(self):
        self.mode = "normal"
        self.running = False
        self._idx = 0
        self.event_log = deque(maxlen=50)
        self._df = None

    def _load_dataset(self):
        """Load and cache the NASA discharge dataset."""
        if self._df is None:
            cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Capacity', 'id_cycle']
            self._df = pd.read_csv(DATASET_PATH, usecols=cols)
            self._log("DATA", f"Loaded NASA dataset: {len(self._df):,} records from discharge.csv")

    def _log(self, tag, msg):
        ts = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{ts}] [{tag}] {msg}")

    def start(self):
        self._load_dataset()
        self.running = True
        self.mode = "normal"
        # Start at mid-dataset (cycle ~80+) so ML models have meaningful context
        self._idx = len(self._df) // 3
        self._log("SYSTEM", "Simulator ONLINE. Replaying real NASA battery telemetry.")

    def set_mode(self, mode):
        prev = self.mode
        self.mode = mode
        desc = self.FAULT_DESCRIPTIONS.get(mode, self.FAULT_DESCRIPTIONS["normal"])

        if mode == "normal" and prev != "normal":
            self._log("RESTORE", "All faults cleared. Returning to real dataset readings.")
        elif mode != "normal":
            self._log("FAULT", f"{desc['title']}")
            self._log("FAULT", f"{desc['detail'][:120]}...")
            self._log("EXPECT", desc["expected"])

    def get_fault_panel(self):
        """Returns the current fault description dict for the UI panel."""
        return self.FAULT_DESCRIPTIONS.get(self.mode, self.FAULT_DESCRIPTIONS["normal"])

    def generate_reading(self):
        """Get next reading: real data for normal, modified data for faults."""
        if not self.running or self._df is None:
            return None

        # Get real row from dataset (cycle through)
        row = self._df.iloc[self._idx % len(self._df)]
        self._idx += 1

        v = float(row['Voltage_measured'])
        c = float(row['Current_measured'])
        t = float(row['Temperature_measured'])
        cap = float(row['Capacity'])
        cyc = int(row['id_cycle'])

        # Apply INSTANT fault modifications to real readings
        if self.mode == "overheat":
            t = 75.0 + np.random.uniform(0, 10)  # 75-85°C (real: 22-42°C)
            v = max(2.0, v - 1.2)                 # voltage drops under thermal stress
            c = c - 0.5 + np.random.uniform(-0.3, 0.3)  # current draws more erratically
            cyc = max(cyc, 120)                   # simulate aged cell under thermal stress

        elif self.mode == "voltage_sag":
            v = 0.5 + np.random.uniform(0, 1.0)   # 0.5-1.5V (real min: 1.7V)
            cap = max(0.5, cap * 0.4)              # capacity collapses
            cyc = max(cyc, 150)                    # deep discharge = end of life

        elif self.mode == "unstable":
            c = np.random.uniform(-8.0, 8.0)       # wild swings (real: -2A steady)
            t = t + abs(c) * 1.5                   # heating from erratic current
            cyc = max(cyc, 130)                    # instability = degraded cell

        return {
            "voltage": round(v, 3),
            "current": round(c, 3),
            "temperature": round(t, 2),
            "capacity": round(cap, 3),
            "id_cycle": cyc,
        }

    def get_log(self):
        return list(self.event_log)
