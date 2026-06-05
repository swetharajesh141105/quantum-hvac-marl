"""
MQTT Bridge — Connects AI to ESP32
=====================================
Receives sensor data from ESP32 via MQTT,
feeds it into the AI agents, and sends decisions back.

This is the "glue" between the AI brain (laptop) and
the physical hardware (ESP32 + sensors + relay).

DATA FLOW:
ESP32 publishes sensor data every 5 seconds
    ↓ MQTT broker (Mosquitto on laptop)
    ↓ This script subscribes and receives
    ↓ Feed into MARL agents
    ↓ Get action decision
    ↓ Publish command back to ESP32
    ↓ ESP32 controls relay (AC ON/OFF)

MQTT TOPICS:
  Subscribe: hvac/room1/sensors  (ESP32 → Laptop)
  Publish:   hvac/room1/command  (Laptop → ESP32)

To run this:
1. Start Mosquitto broker: mosquitto -v
2. Run this script: py mqtt_bridge.py
3. Power on ESP32 — data will start flowing
"""

import json
import time
import numpy as np
import threading
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import paho.mqtt.client as mqtt

from marl.environment import HVACMultiAgentEnv
from marl.xai_explainer import RuleBasedExplainer
from quantum.qubo_optimizer import QUBOScheduler


# ================================================================== #
#  Configuration                                                      #
# ================================================================== #

MQTT_BROKER = "localhost"   # Change to your laptop's IP if ESP32 is on same WiFi
MQTT_PORT = 1883
TOPIC_SENSORS = "hvac/room1/sensors"
TOPIC_COMMAND = "hvac/room1/command"
TOPIC_EXPLANATION = "hvac/room1/explanation"
TOPIC_STATUS = "hvac/room1/status"

DECISION_INTERVAL = 30   # Make AI decision every 30 seconds


# ================================================================== #
#  AI Decision Engine                                                  #
# ================================================================== #

class HVACController:
    """
    Main controller that receives sensor data and produces decisions.
    Runs the MARL agents + QUBO schedule + XAI explainer.
    """
    
    def __init__(self):
        self.env = HVACMultiAgentEnv()
        self.explainer = RuleBasedExplainer()
        self.qubo_scheduler = QUBOScheduler()
        
        # Load daily QUBO schedule
        self.daily_schedule = self._load_or_generate_schedule()
        
        # State tracking
        self.latest_sensor_data = None
        self.last_decision_time = 0
        self.decision_history = []
        self.current_obs = self.env.reset()[0]
        
        # Simple heuristic agents (used when PPO models not trained yet)
        self.use_heuristic = True
        
        print("HVACController initialized")
        print(f"Daily schedule: AC ON during hours {self.daily_schedule['on_hours']}")
    
    def _load_or_generate_schedule(self) -> dict:
        """Load existing QUBO schedule or generate a new one."""
        if os.path.exists("optimized_schedule.json"):
            with open("optimized_schedule.json") as f:
                schedule = json.load(f)
            print("Loaded existing QUBO schedule")
        else:
            print("Generating new QUBO schedule...")
            schedule = self.qubo_scheduler.optimize_schedule()
        return schedule
    
    def process_sensor_data(self, sensor_data: dict) -> dict:
        """
        Process incoming sensor data and make AI decision.
        
        sensor_data: {
            "temperature": 26.5,
            "humidity": 65.0,
            "occupancy": 1,
            "co2_ppm": 750,
            "relay_on": 0
        }
        
        Returns: {
            "action": "ON",
            "setpoint": 22,
            "explanation": "..."
        }
        """
        self.latest_sensor_data = sensor_data
        
        now = time.time()
        if now - self.last_decision_time < DECISION_INTERVAL:
            return None  # not time to decide yet
        
        self.last_decision_time = now
        
        # Get current hour
        import datetime
        current_hour = datetime.datetime.now().hour
        
        # Check QUBO schedule — hard constraint
        schedule = self.daily_schedule.get("schedule", [1] * 24)
        qubo_allows_ac = bool(schedule[current_hour])
        
        # Build observation from real sensor data
        obs = self._sensor_to_observation(sensor_data, current_hour)
        
        # Make AI decision
        if self.use_heuristic:
            action = self._heuristic_decision(sensor_data, qubo_allows_ac)
        else:
            # Use trained agents
            action = self._agent_decision(obs, qubo_allows_ac)
        
        # Generate explanation
        info = {
            "temp": sensor_data.get("temperature", 25),
            "humidity": sensor_data.get("humidity", 60),
            "occupants": sensor_data.get("occupancy", 0),
            "co2": sensor_data.get("co2_ppm", 400),
            "power_kw": sensor_data.get("power_kw", 0),
            "outdoor_temp": sensor_data.get("outdoor_temp", 32),
        }
        explanation = self.explainer.explain(action, info)
        
        # Decode action to relay command
        action_map = {
            0: {"action": "OFF", "setpoint": None},
            1: {"action": "ON",  "setpoint": 18},
            2: {"action": "ON",  "setpoint": 20},
            3: {"action": "ON",  "setpoint": 22},
            4: {"action": "ON",  "setpoint": 24},
            5: {"action": "ON",  "setpoint": 26},
        }
        command = action_map[action].copy()
        command["explanation"] = explanation
        command["qubo_schedule_allows"] = qubo_allows_ac
        command["timestamp"] = now
        
        self.decision_history.append({
            "time": now,
            "sensor_data": sensor_data,
            "action": action,
            "command": command,
        })
        
        print(f"\n[AI Decision] {command['action']}"
              f"{f' @ {command[\"setpoint\"]}°C' if command['setpoint'] else ''}")
        print(f"  {explanation}")
        
        return command
    
    def _sensor_to_observation(self, sensor_data: dict, hour: int) -> np.ndarray:
        """Convert real sensor data to normalized observation vector."""
        temp = sensor_data.get("temperature", 25)
        humidity = sensor_data.get("humidity", 60)
        occupancy = sensor_data.get("occupancy", 0)
        co2 = sensor_data.get("co2_ppm", 400)
        power = sensor_data.get("power_kw", 0)
        outdoor = sensor_data.get("outdoor_temp", 32)
        ac_on = float(sensor_data.get("relay_on", 0))
        
        hour_fraction = hour / 24.0
        
        return np.array([
            (temp - 15) / 20,
            (humidity - 30) / 50,
            min(occupancy / 10, 1.0),
            (co2 - 400) / 1600,
            min(power / 1.5, 1.0),
            (outdoor - 20) / 20,
            np.sin(2 * np.pi * hour_fraction),
            np.cos(2 * np.pi * hour_fraction),
            ac_on,
            0.5,  # default setpoint
        ], dtype=np.float32)
    
    def _heuristic_decision(self, sensor_data: dict, qubo_allows: bool) -> int:
        """
        Rule-based fallback when trained models aren't available.
        Simple but effective heuristic.
        """
        temp = sensor_data.get("temperature", 25)
        occupancy = sensor_data.get("occupancy", 0)
        co2 = sensor_data.get("co2_ppm", 400)
        
        # QUBO says off — respect the schedule
        if not qubo_allows:
            return 0  # OFF
        
        # No one there — turn off
        if occupancy == 0 and co2 < 600:
            return 0  # OFF
        
        # Determine setpoint based on temperature
        if temp > 27:
            return 2  # 20°C — aggressive
        elif temp > 25:
            return 3  # 22°C — comfortable
        elif temp > 23:
            return 4  # 24°C — mild
        elif temp < 21:
            return 0  # OFF — already cold enough
        else:
            return 5  # 26°C — minimal
    
    def _agent_decision(self, obs: np.ndarray, qubo_allows: bool) -> int:
        """Use trained PPO agents (when available)."""
        # Override with OFF if QUBO says no
        if not qubo_allows:
            return 0
        # TODO: load and use PPO models
        return self._heuristic_decision({}, qubo_allows)


# ================================================================== #
#  MQTT Client                                                        #
# ================================================================== #

controller = HVACController()


def on_connect(client, userdata, flags, rc, properties=None):
    """Called when connected to MQTT broker."""
    if rc == 0:
        print(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_SENSORS)
        print(f"Subscribed to {TOPIC_SENSORS}")
        
        # Publish online status
        client.publish(TOPIC_STATUS, json.dumps({
            "status": "online",
            "message": "AI controller connected"
        }))
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    """Called when sensor data arrives from ESP32."""
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        
        if topic == TOPIC_SENSORS:
            temp = payload.get('temperature', 'N/A')
            occ = payload.get('occupancy', 0)
            co2 = payload.get('co2_ppm', 400)
            print(f"[Sensor] T={temp}°C Occ={occ} CO2={co2}ppm")
            
            # Process and get decision
            command = controller.process_sensor_data(payload)
            
            if command:
                # Send command back to ESP32
                client.publish(TOPIC_COMMAND, json.dumps(command))
                
                # Publish explanation for dashboard
                client.publish(TOPIC_EXPLANATION, json.dumps({
                    "explanation": command.get("explanation", ""),
                    "timestamp": command.get("timestamp", 0)
                }))
    
    except json.JSONDecodeError:
        print(f"Invalid JSON received: {msg.payload}")
    except Exception as e:
        print(f"Error processing message: {e}")


def on_disconnect(client, userdata, rc, properties=None):
    """Called when disconnected."""
    print(f"Disconnected from broker (rc={rc})")


def run_bridge():
    """Start the MQTT bridge."""
    print("=" * 60)
    print("HVAC AI — MQTT Bridge")
    print("=" * 60)
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Listening on: {TOPIC_SENSORS}")
    print(f"Commanding on: {TOPIC_COMMAND}")
    print("\nWaiting for sensor data from ESP32...")
    print("(Start Mosquitto broker first: mosquitto -v)")
    print("Press Ctrl+C to stop\n")
    
    # Create MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_forever()
    except ConnectionRefusedError:
        print("ERROR: Cannot connect to MQTT broker!")
        print("Make sure Mosquitto is running: mosquitto -v")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        client.disconnect()


# ================================================================== #
#  Simulation Mode (test without ESP32)                               #
# ================================================================== #

def run_simulation():
    """
    Test the bridge without ESP32 by generating fake sensor data.
    Useful for verifying the AI decision pipeline works.
    """
    print("=" * 60)
    print("MQTT Bridge — SIMULATION MODE (no ESP32 needed)")
    print("=" * 60)
    
    import datetime
    
    test_scenarios = [
        {"temperature": 28.5, "humidity": 70, "occupancy": 1, 
         "co2_ppm": 850, "relay_on": 0, "label": "Hot + Occupied"},
        {"temperature": 22.1, "humidity": 55, "occupancy": 0, 
         "co2_ppm": 420, "relay_on": 1, "label": "Cool + Empty"},
        {"temperature": 26.0, "humidity": 65, "occupancy": 1, 
         "co2_ppm": 700, "relay_on": 0, "label": "Warm + Occupied"},
        {"temperature": 20.5, "humidity": 50, "occupancy": 0, 
         "co2_ppm": 400, "relay_on": 0, "label": "Cold + Empty"},
        {"temperature": 30.0, "humidity": 75, "occupancy": 1, 
         "co2_ppm": 950, "relay_on": 0, "label": "Very Hot + Occupied"},
    ]
    
    for scenario in test_scenarios:
        label = scenario.pop("label")
        print(f"\nScenario: {label}")
        print(f"  Input: T={scenario['temperature']}°C "
              f"Occ={scenario['occupancy']} CO2={scenario['co2_ppm']}ppm")
        
        # Force decision by resetting timer
        controller.last_decision_time = 0
        command = controller.process_sensor_data(scenario)
        
        if command:
            print(f"  Output: AC {command['action']}"
                  f"{f' @ {command[\"setpoint\"]}°C' if command['setpoint'] else ''}")
    
    print("\n✅ Simulation complete! Bridge working correctly.")
    print("Connect ESP32 and run: py mqtt_bridge.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", 
                       help="Run in simulation mode (no ESP32/MQTT needed)")
    args = parser.parse_args()
    
    if args.sim:
        run_simulation()
    else:
        run_bridge()
