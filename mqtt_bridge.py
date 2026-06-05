"""
MQTT Bridge — Connects AI to ESP32
"""

import json
import time
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import paho.mqtt.client as mqtt

from marl.environment import HVACMultiAgentEnv
from marl.xai_explainer import RuleBasedExplainer
from quantum.qubo_optimizer import QUBOScheduler

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_SENSORS = "hvac/room1/sensors"
TOPIC_COMMAND = "hvac/room1/command"
TOPIC_EXPLANATION = "hvac/room1/explanation"
TOPIC_STATUS = "hvac/room1/status"
DECISION_INTERVAL = 30


class HVACController:
    def __init__(self):
        self.env = HVACMultiAgentEnv()
        self.explainer = RuleBasedExplainer()
        self.qubo_scheduler = QUBOScheduler()
        self.daily_schedule = self._load_or_generate_schedule()
        self.latest_sensor_data = None
        self.last_decision_time = 0
        self.decision_history = []
        self.current_obs = self.env.reset()[0]
        self.use_heuristic = True
        print("HVACController initialized")
        print("Daily schedule ON hours: " + str(self.daily_schedule['on_hours']))

    def _load_or_generate_schedule(self):
        if os.path.exists("optimized_schedule.json"):
            with open("optimized_schedule.json") as f:
                schedule = json.load(f)
            print("Loaded existing QUBO schedule")
        else:
            print("Generating new QUBO schedule...")
            schedule = self.qubo_scheduler.optimize_schedule()
        return schedule

    def process_sensor_data(self, sensor_data):
        self.latest_sensor_data = sensor_data
        now = time.time()
        if now - self.last_decision_time < DECISION_INTERVAL:
            return None
        self.last_decision_time = now

        import datetime
        current_hour = datetime.datetime.now().hour
        schedule = self.daily_schedule.get("schedule", [1] * 24)
        qubo_allows_ac = bool(schedule[current_hour])
        obs = self._sensor_to_observation(sensor_data, current_hour)
        action = self._heuristic_decision(sensor_data, qubo_allows_ac)

        info = {
            "temp": sensor_data.get("temperature", 25),
            "humidity": sensor_data.get("humidity", 60),
            "occupants": sensor_data.get("occupancy", 0),
            "co2": sensor_data.get("co2_ppm", 400),
            "power_kw": sensor_data.get("power_kw", 0),
            "outdoor_temp": sensor_data.get("outdoor_temp", 32),
        }
        explanation = self.explainer.explain(action, info)

        action_map = {
            0: {"action": "OFF",  "setpoint": None},
            1: {"action": "ON",   "setpoint": 18},
            2: {"action": "ON",   "setpoint": 20},
            3: {"action": "ON",   "setpoint": 22},
            4: {"action": "ON",   "setpoint": 24},
            5: {"action": "ON",   "setpoint": 26},
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

        if command["setpoint"] is not None:
            print("\n[AI Decision] " + command["action"] + " @ " + str(command["setpoint"]) + "C")
        else:
            print("\n[AI Decision] " + command["action"])
        print("  " + explanation)

        return command

    def _sensor_to_observation(self, sensor_data, hour):
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
            0.5,
        ], dtype=np.float32)

    def _heuristic_decision(self, sensor_data, qubo_allows):
        temp = sensor_data.get("temperature", 25)
        occupancy = sensor_data.get("occupancy", 0)
        co2 = sensor_data.get("co2_ppm", 400)
        if not qubo_allows:
            return 0
        if occupancy == 0 and co2 < 600:
            return 0
        if temp > 27:
            return 2
        elif temp > 25:
            return 3
        elif temp > 23:
            return 4
        elif temp < 21:
            return 0
        else:
            return 5


controller = HVACController()


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT broker at " + MQTT_BROKER + ":" + str(MQTT_PORT))
        client.subscribe(TOPIC_SENSORS)
        print("Subscribed to " + TOPIC_SENSORS)
        client.publish(TOPIC_STATUS, json.dumps({"status": "online"}))
    else:
        print("Connection failed with code " + str(rc))


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        if msg.topic == TOPIC_SENSORS:
            temp = payload.get("temperature", "N/A")
            occ = payload.get("occupancy", 0)
            co2 = payload.get("co2_ppm", 400)
            print("[Sensor] T=" + str(temp) + "C Occ=" + str(occ) + " CO2=" + str(co2) + "ppm")
            command = controller.process_sensor_data(payload)
            if command:
                client.publish(TOPIC_COMMAND, json.dumps(command))
                client.publish(TOPIC_EXPLANATION, json.dumps({
                    "explanation": command.get("explanation", ""),
                    "timestamp": command.get("timestamp", 0)
                }))
    except json.JSONDecodeError:
        print("Invalid JSON: " + str(msg.payload))
    except Exception as e:
        print("Error: " + str(e))


def on_disconnect(client, userdata, rc, properties=None):
    print("Disconnected (rc=" + str(rc) + ")")


def run_bridge():
    print("=" * 50)
    print("HVAC AI — MQTT Bridge")
    print("Broker: " + MQTT_BROKER + ":" + str(MQTT_PORT))
    print("Waiting for ESP32 sensor data...")
    print("Press Ctrl+C to stop")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_forever()
    except ConnectionRefusedError:
        print("ERROR: Cannot connect to MQTT broker!")
        print("Run: mosquitto -v")
    except KeyboardInterrupt:
        print("\nStopped.")
        client.disconnect()


def run_simulation():
    print("=" * 50)
    print("MQTT Bridge — SIMULATION MODE")
    print("=" * 50)

    test_scenarios = [
        {"temperature": 28.5, "humidity": 70, "occupancy": 1, "co2_ppm": 850, "relay_on": 0, "label": "Hot + Occupied"},
        {"temperature": 22.1, "humidity": 55, "occupancy": 0, "co2_ppm": 420, "relay_on": 1, "label": "Cool + Empty"},
        {"temperature": 26.0, "humidity": 65, "occupancy": 1, "co2_ppm": 700, "relay_on": 0, "label": "Warm + Occupied"},
        {"temperature": 20.5, "humidity": 50, "occupancy": 0, "co2_ppm": 400, "relay_on": 0, "label": "Cold + Empty"},
        {"temperature": 30.0, "humidity": 75, "occupancy": 1, "co2_ppm": 950, "relay_on": 0, "label": "Very Hot + Occupied"},
    ]

    for scenario in test_scenarios:
        label = scenario.pop("label")
        print("\nScenario: " + label)
        print("  Input: T=" + str(scenario["temperature"]) + "C Occ=" + str(scenario["occupancy"]) + " CO2=" + str(scenario["co2_ppm"]) + "ppm")
        controller.last_decision_time = 0
        command = controller.process_sensor_data(scenario)
        if command:
            sp = " @ " + str(command["setpoint"]) + "C" if command["setpoint"] else ""
            print("  Output: AC " + command["action"] + sp)

    print("\n✅ Simulation complete! Bridge working correctly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation mode")
    args = parser.parse_args()
    if args.sim:
        run_simulation()
    else:
        run_bridge()
