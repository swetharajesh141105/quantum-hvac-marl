# Quantum-Inspired Federated MARL for Smart HVAC
Setup: pip install -r requirements.txt, then cd digital_twin
Run:
- python3 run_full_pipeline.py     (single-building: MARL + QUBO + XAI)
- python3 multi_building_qubo.py   (3-building QUBO grid coordination)
- python3 hvac_audit_full.py       (30-trial statistical audit)

## Digital-twin QUBO planner: results (simulation)
| Feature | Script | Result |
|---|---|---|
| Tariff-aware pre-cooling | tariff_qubo_demo.py | ~12.8% lower cost vs thermostat (30 days, assumed tariff) |
| Grid-event re-planning | grid_event.py | Violations 2.35 to 0.90 h/day vs morning plan (cost higher) |
| Probe-and-learn calibration | calib_dayahead.py | Violations 0.90 to 0.10 h/day (cost higher) |

Limits: simulated room, assumed tariff, single building. Self-calibration gave no extra benefit when hourly re-planning was already on.
