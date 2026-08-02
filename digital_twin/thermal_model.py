"""
Thermal Physics Core
Simulates room temperature using a simple RC (resistor-capacitor) thermal model.
"""

class RoomThermalModel:
    def __init__(self, room_params: dict):
        """
        room_params expected keys (matches config.yaml from Person C):
            thermal_mass: float      # how slowly room temp changes (higher = slower)
            insulation_resistance: float  # how well room resists outdoor heat
            ac_cooling_power: float  # degrees/hour AC can remove
            occupancy_heat_gain: float  # degrees/hour added per occupant
        """
        self.thermal_mass = room_params.get("thermal_mass", 5.0)
        self.insulation_resistance = room_params.get("insulation_resistance", 2.0)
        self.ac_cooling_power = room_params.get("ac_cooling_power", 3.0)
        self.occupancy_heat_gain = room_params.get("occupancy_heat_gain", 0.5)

    def step(self, current_temp, outdoor_temp, ac_state, occupancy, dt):
        """
        current_temp: float (current room temp, °C)
        outdoor_temp: float (°C)
        ac_state: 0 or 1 (AC off/on)
        occupancy: 0 or 1 (room empty/occupied)
        dt: float (timestep in hours, e.g. 1/60 for 1 minute)

        Returns: new_temp (float)
        """
        # heat flows into room from outside (Newton's law of cooling)
        heat_gain = (outdoor_temp - current_temp) / self.insulation_resistance

        # occupancy adds heat
        heat_gain += occupancy * self.occupancy_heat_gain

        # AC removes heat if ON
        heat_loss = ac_state * self.ac_cooling_power

        # net temperature change
        d_temp = (heat_gain - heat_loss) / self.thermal_mass

        new_temp = current_temp + d_temp * dt
        return new_temp


if __name__ == "__main__":
    # quick manual sanity check
    model = RoomThermalModel({
        "thermal_mass": 5.0,
        "insulation_resistance": 2.0,
        "ac_cooling_power": 3.0,
        "occupancy_heat_gain": 0.5,
    })
    temp = 28.0
    for hour in range(5):
        temp = model.step(temp, outdoor_temp=35.0, ac_state=1, occupancy=1, dt=1.0)
        print(f"Hour {hour+1}: Room temp = {temp:.2f} °C")