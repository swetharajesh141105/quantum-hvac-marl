# ESP32 Sensor Node — MicroPython
# ===================================
# Flash this onto your ESP32 using Thonny IDE or ampy tool.
#
# HARDWARE CONNECTIONS:
#   DHT22   → GPIO 4  (Data pin, with 10kΩ pull-up to 3.3V)
#   PIR     → GPIO 14 (Output pin — HIGH when motion detected)
#   MQ-135  → GPIO 34 (Analog input — ADC for CO2 estimation)
#   Relay   → GPIO 26 (HIGH = relay ON = AC ON)
#   LED     → GPIO 2  (Built-in LED = status indicator)
#
# MQTT TOPICS:
#   Publish:  hvac/room1/temperature
#             hvac/room1/humidity
#             hvac/room1/occupancy
#             hvac/room1/co2
#   Subscribe: hvac/room1/command  (receives ON/OFF from AI)
#
# PROTOCOL EXPLANATION:
#   MQTT (Message Queuing Telemetry Transport) is a lightweight
#   publish-subscribe messaging protocol designed for IoT.
#   Unlike HTTP (request-response), MQTT uses a broker model:
#   - Sensors PUBLISH data to topics
#   - AI server SUBSCRIBES to topics to receive data
#   - Server PUBLISHES commands → ESP32 subscribes and acts
#   This is event-driven and uses very little bandwidth/power.

import machine
import network
import time
import json
import dht

# --- MicroPython MQTT library (umqtt.simple) ---
# Install via: import upip; upip.install('micropython-umqtt.simple')
try:
    from umqtt.simple import MQTTClient
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("MQTT library not installed. Running in debug mode.")

# ================================================================== #
#  CONFIGURATION — Edit these before flashing                        #
# ================================================================== #

WIFI_SSID = "YOUR_WIFI_NAME"        # Your WiFi network name
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD" # Your WiFi password
MQTT_BROKER = "192.168.1.100"        # IP of your laptop (running broker)
MQTT_PORT = 1883
DEVICE_ID = "room1"
ROOM_ID = f"hvac/{DEVICE_ID}"

# GPIO Pin assignments
PIN_DHT22 = 4
PIN_PIR = 14
PIN_CO2_ADC = 34    # ADC1 channel (GPIO 34-39 are input-only on ESP32)
PIN_RELAY = 26
PIN_LED = 2

# Publishing interval
PUBLISH_INTERVAL_MS = 5000   # every 5 seconds

# ================================================================== #
#  Hardware Setup                                                     #
# ================================================================== #

# DHT22 temperature & humidity sensor
dht_sensor = dht.DHT22(machine.Pin(PIN_DHT22))

# PIR motion sensor (digital HIGH/LOW)
pir_sensor = machine.Pin(PIN_PIR, machine.Pin.IN)

# CO2 sensor (analog — reads voltage proportional to CO2 ppm)
co2_adc = machine.ADC(machine.Pin(PIN_CO2_ADC))
co2_adc.atten(machine.ADC.ATTN_11DB)   # 0–3.6V range

# Relay module (controls AC)
relay = machine.Pin(PIN_RELAY, machine.Pin.OUT)
relay.value(0)  # Start with relay OFF

# Status LED
led = machine.Pin(PIN_LED, machine.Pin.OUT)


# ================================================================== #
#  WiFi Connection                                                    #
# ================================================================== #

def connect_wifi():
    """Connect to WiFi with retry logic."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        return wlan.ifconfig()[0]

    print(f"Connecting to WiFi: {WIFI_SSID}")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    timeout = 30  # seconds
    start = time.time()
    while not wlan.isconnected():
        if time.time() - start > timeout:
            raise OSError("WiFi connection timed out!")
        led.value(not led.value())  # blink LED while connecting
        time.sleep(0.5)

    ip = wlan.ifconfig()[0]
    print(f"WiFi connected! IP: {ip}")
    led.value(1)  # solid LED = connected
    return ip


# ================================================================== #
#  Sensor Reading Functions                                           #
# ================================================================== #

def read_dht22():
    """
    Read temperature and humidity from DHT22.
    Returns (temperature_C, humidity_percent) or (None, None) on error.
    
    DHT22 specs:
      Temperature: -40 to +80°C, ±0.5°C accuracy
      Humidity: 0-100% RH, ±2% accuracy
      Sampling rate: max 1Hz (don't read faster than once per second)
    """
    try:
        dht_sensor.measure()
        time.sleep_ms(200)  # DHT22 needs time after measure()
        return dht_sensor.temperature(), dht_sensor.humidity()
    except OSError as e:
        print(f"DHT22 error: {e}")
        return None, None


def read_pir():
    """
    Read PIR motion sensor.
    Returns 1 if motion detected, 0 if no motion.
    
    PIR (Passive Infrared) detects body heat movement.
    Has a ~30s hold time — stays HIGH for 30s after last motion.
    """
    return int(pir_sensor.value())


def read_co2_ppm():
    """
    Estimate CO2 in ppm from MQ-135 analog reading.
    
    MQ-135 raw output: 0-4095 (12-bit ADC)
    Calibration: Needs Rs/Ro calibration in clean air (400ppm baseline).
    This is a simplified linear estimate — production use needs full
    Steinhart-Hart equation calibration.
    
    For accurate CO2 (not estimated): use SCD40/SCD30 I2C sensor.
    """
    raw = co2_adc.read()
    # Simplified piecewise linear calibration
    # Raw 0→ ~400 ppm (clean air)
    # Raw 4095 → ~2000 ppm (very high CO2)
    ppm = 400 + (raw / 4095) * 1600
    return int(ppm)


# ================================================================== #
#  MQTT Command Handler (receives AI decisions)                       #
# ================================================================== #

def on_message(topic, msg):
    """
    Called when AI server publishes a command.
    
    Expected message format: JSON
    {"action": "ON", "setpoint": 22} or {"action": "OFF"}
    """
    try:
        topic_str = topic.decode()
        payload = json.loads(msg.decode())
        print(f"Received command: {payload}")

        action = payload.get("action", "OFF")
        setpoint = payload.get("setpoint", 24)

        if action == "ON":
            relay.value(1)
            led.value(1)
            print(f"  → AC turned ON (setpoint: {setpoint}°C)")
        else:
            relay.value(0)
            led.value(0)
            print(f"  → AC turned OFF")

    except Exception as e:
        print(f"Command parse error: {e}")


# ================================================================== #
#  Main Loop                                                          #
# ================================================================== #

def main():
    """Main execution loop."""
    print("=" * 40)
    print("Quantum HVAC — ESP32 Sensor Node")
    print(f"Device ID: {DEVICE_ID}")
    print("=" * 40)

    # Connect WiFi
    try:
        ip = connect_wifi()
    except OSError as e:
        print(f"WiFi failed: {e}. Running in offline mode.")
        ip = None

    # Setup MQTT
    client = None
    if MQTT_AVAILABLE and ip:
        try:
            client = MQTTClient(
                client_id=f"esp32_{DEVICE_ID}",
                server=MQTT_BROKER,
                port=MQTT_PORT,
                keepalive=60
            )
            client.set_callback(on_message)
            client.connect()
            client.subscribe(f"{ROOM_ID}/command")
            print(f"MQTT connected to broker at {MQTT_BROKER}")
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            client = None

    # Sensor reading loop
    print("\nStarting sensor loop (every 5 seconds)...")
    last_publish = 0

    while True:
        try:
            now = time.ticks_ms()

            # Check for incoming MQTT messages (non-blocking)
            if client:
                client.check_msg()

            # Publish sensor data every 5 seconds
            if time.ticks_diff(now, last_publish) >= PUBLISH_INTERVAL_MS:
                temp, hum = read_dht22()
                occupancy = read_pir()
                co2 = read_co2_ppm()
                relay_state = relay.value()

                # Build sensor payload
                payload = {
                    "device_id": DEVICE_ID,
                    "timestamp": time.time(),
                    "temperature": temp,
                    "humidity": hum,
                    "occupancy": occupancy,
                    "co2_ppm": co2,
                    "relay_on": relay_state,
                }

                print(f"T={temp}°C H={hum}% Occ={occupancy} CO2={co2}ppm AC={'ON' if relay_state else 'OFF'}")

                # Publish to MQTT broker
                if client:
                    payload_json = json.dumps(payload).encode()
                    client.publish(f"{ROOM_ID}/sensors", payload_json)
                    led.value(not led.value())  # blink = published

                last_publish = now

            time.sleep_ms(100)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)


# ================================================================== #
#  Run                                                                #
# ================================================================== #
if __name__ == "__main__":
    main()
