# Quantum-Enhanced Smart HVAC System

> B.Tech Final Year Project — VIT Vellore | SCOPE | IoT Specialization  
> Team: 3 members | Guide: Dr.Jayashree 

## What This Project Does

A smart building HVAC (Air Conditioning) controller that uses AI to reduce energy consumption while maintaining comfort. The system learns occupancy patterns, predicts when rooms will be used, and makes optimal cooling decisions.

**Key Results:**
- **46.2% energy savings** vs always-on baseline (MARL agents)
- **69.7% energy savings** vs always-on baseline (QUBO schedule)
- Privacy-preserving multi-building learning (Federated Learning)

---

## System Architecture

```
ESP32 (Sensor Node)          Laptop (AI Brain)
──────────────────           ─────────────────────────────
DHT22 → Temperature    ──►   MARL Agents (3 cooperative)
PIR   → Occupancy      ──►   QUBO Quantum Optimizer
CO2   → Air Quality    ──►   LSTM Occupancy Predictor
                             Federated Learning Server
                       ◄──   AI Decision (ON/OFF/Setpoint)
Relay → AC Control
```

---

## Components

### 1. Digital Twin (`digital_twin/thermal_model.py`)
Physics-based simulation of room thermal dynamics. Based on Newton's Law of Cooling. Used to safely train RL agents before real hardware deployment.

### 2. Multi-Agent RL (`marl/`)
3 cooperative PPO agents trained simultaneously:
- **Energy Agent** — minimizes power consumption
- **Comfort Agent** — maintains 22°C when occupied  
- **Carbon Agent** — prefers clean energy hours (solar peak)

### 3. QUBO Optimizer (`quantum/qubo_optimizer.py`)
Quantum-inspired scheduling using Simulated Annealing. Formulates 24-hour AC scheduling as a QUBO (Quadratic Unconstrained Binary Optimization) problem. No real quantum hardware needed — uses classical simulator.

### 4. Federated Learning (`federated/fl_training.py`)
Privacy-preserving collaborative learning across multiple buildings. Each building trains locally, shares only model weights (never raw data). Uses FedAvg aggregation algorithm.

### 5. LSTM Occupancy Predictor (`marl/lstm_occupancy.py`)
Predicts room occupancy 30 minutes ahead using sensor history. Enables pre-cooling before people arrive.

### 6. Explainable AI (`marl/xai_explainer.py`)
Explains every AI decision in natural language:
> "AC turned OFF because: room is empty (no one to comfort); temperature 22.1°C is already comfortable"

### 7. ESP32 Sensor Node (`esp32/sensor_node.py`)
MicroPython code for ESP32. Reads DHT22, PIR, CO2 sensors and publishes via MQTT. Receives AI commands and controls relay.

### 8. MQTT Bridge (`mqtt_bridge.py`)
Connects AI brain to ESP32 hardware via MQTT protocol.

---

## Quick Start

### Setup
```bash
git clone https://github.com/swetharajesh141105/quantum-hvac-marl.git
cd quantum-hvac-marl
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### Run (Software Only — No Hardware Needed)
```bash
# Test Digital Twin
py digital_twin/thermal_model.py

# Test MARL Environment  
py marl/environment.py

# Train MARL Agents (basic)
py marl/train_agents.py

# Train with PPO (better)
py marl/train_sb3.py

# Run QUBO Optimizer
py quantum/qubo_optimizer.py

# Run Federated Learning
py federated/fl_training.py

# Train LSTM Predictor
py marl/lstm_occupancy.py

# Test XAI Explainer
py marl/xai_explainer.py

# Test MQTT Bridge (no ESP32)
py mqtt_bridge.py --sim
```

---

## Hardware Required
- ESP32 development board
- DHT22 temperature + humidity sensor
- PIR motion sensor
- MQ-135 CO2 sensor (or SCD40 for accuracy)
- 5V relay module
- Desktop fan (for demo) or AC unit

## Hardware Wiring
| Sensor | ESP32 Pin |
|--------|-----------|
| DHT22 DATA | GPIO 4 |
| PIR OUT | GPIO 14 |
| CO2 ADC | GPIO 34 |
| Relay IN | GPIO 26 |

---

## Tech Stack
- **Python 3.12** — core language
- **PyTorch + Stable-Baselines3** — RL training (PPO)
- **Qiskit** — quantum-inspired optimization
- **Flower (flwr)** — federated learning
- **MicroPython** — ESP32 firmware
- **paho-mqtt** — MQTT communication
- **Grafana + Node-RED** — dashboard

---

## Project Structure
```
quantum-hvac-marl/
├── digital_twin/
│   └── thermal_model.py      # Room physics simulation
├── marl/
│   ├── environment.py        # Multi-agent RL environment
│   ├── train_agents.py       # Basic training
│   ├── train_sb3.py          # PPO training (SB3)
│   ├── lstm_occupancy.py     # Occupancy prediction
│   └── xai_explainer.py      # Explainable AI
├── quantum/
│   └── qubo_optimizer.py     # QUBO scheduling
├── federated/
│   └── fl_training.py        # Federated learning
├── esp32/
│   └── sensor_node.py        # MicroPython ESP32 code
├── dashboard/                # Grafana/Node-RED configs
├── docs/                     # Report, paper drafts
├── mqtt_bridge.py            # AI ↔ ESP32 connector
└── requirements.txt
```

---

## Team
| Member | Role |
|--------|------|
| Swetha Rajesh | Digital Twin, MARL, QUBO, FL |
| Dharshinikesan K S | Hardware, ESP32, Dashboard |
| Harshavardhini | Integration, Testing, Paper |

**VIT Vellore | BCSE — IoT Specialization | 2025–26**
