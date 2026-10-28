# Quantum-Inspired Federated MARL for Smart HVAC

Simulation study: a digital-twin room, reinforcement-learning agents, federated averaging across buildings, and QUBO planning (simulated annealing) with explanations for each decision.

## Setup
pip install -r requirements.txt
cd digital_twin

## Run
| Command | What it shows |
|---|---|
| `python3 run_full_pipeline.py` | Single building: agents + QUBO + explanations |
| `python3 multi_building_qubo.py` | 3 buildings sharing a grid limit |
| `python3 tariff_qubo_demo.py` | Tariff-aware planning, 30 days (assumed tariff) |
| `python3 demo_day.py` | One day, hour by hour |
| `python3 hvac_audit_full.py` | 30-trial statistical audit |
| `python3 extra_experiment.py` | Real Vellore weather + TNERC-style tariff (30-60 min; saved in `extra_out.txt`) |

## Results (simulation only)
| Feature | Script | Result |
|---|---|---|
| Single-building control | `run_full_pipeline.py` | 79.2% fewer AC hours than always-on; QUBO changed 0/24 hours |
| Multi-building coordination | `multi_building_qubo.py` | Grid-limit violations 3/24 to 0/24; buildings keep 80%, 86%, 67% of their cooling hours |
| Tariff-aware pre-cooling | `tariff_qubo_demo.py` | ~12.8% lower cost vs thermostat (30 days, assumed tariff), fewer comfort violations |
| Real weather + TNERC multipliers | `extra_experiment.py` | 6.3% (+/-3.0) lower cost vs thermostat, fewer uncomfortable hours (0.70 vs 0.97 h/day) |
| Break-even | `extra_experiment.py` | Saving rises from 3.3% to 23.3% as the peak/night price ratio grows from 1.0 to 6.0 |
| Grid-event re-planning | `grid_event.py` | Violations 2.35 to 0.90 h/day vs morning plan (cost higher) |
| Probe-and-learn calibration | `calib_dayahead.py` | Violations 0.90 to 0.10 h/day (cost higher) |

## Data
- Weather: Open-Meteo historical hourly data for Vellore, April 2025 (gridded reanalysis, not a station reading).
- Tariff: TNERC-style multipliers (peak 1.25x at 06-10 and 18-22, night 0.95x). Check against the tariff order before citing.

## Limits
- Simulated room, not checked against a real building; one month of weather; assumed occupancy.
- Tariff in `tariff_qubo_demo.py` is assumed. Under the TNERC multipliers a flat tariff already saves 4.7%, so price shifting adds only about 1.6 points.
- Brute force beats QUBO at this size (10.0% vs 6.3%); QUBO matters at larger scale.
- With a 2 C forecast error, brute force breaks comfort (1.73 uncomfortable h/day) while QUBO stays near baseline (1.00).
- Energy and carbon agents collapsed to always-OFF in our ablation; the comfort agent plus a safety rule does the work. Federated agents match local ones in accuracy; the benefit is privacy.
- Self-calibration gave no extra benefit when hourly re-planning was already on.
- Hardware (ESP32/MQTT) is not built; this repo is simulation only.
