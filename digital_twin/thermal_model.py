class RoomThermalModel:
    def __init__(self, room_params: dict):
        self.thermal_mass = room_params.get("thermal_mass", 5.0)
        self.insulation_resistance = room_params.get("insulation_resistance", 2.0)
        self.ac_cooling_power = room_params.get("ac_cooling_power", 3.0)
        self.occupancy_heat_gain = room_params.get("occupancy_heat_gain", 0.5)

    def step(self, current_temp, outdoor_temp, ac_state, occupancy, dt):
        heat_gain = (outdoor_temp - current_temp) / self.insulation_resistance
        heat_gain += occupancy * self.occupancy_heat_gain
        heat_loss = ac_state * self.ac_cooling_power
        d_temp = (heat_gain - heat_loss) / self.thermal_mass
        return current_temp + d_temp * dt

if __name__ == "__main__":
    model = RoomThermalModel({"thermal_mass":5.0, "insulation_resistance":2.0, 
                              "ac_cooling_power":3.0, "occupancy_heat_gain":0.5})
    temp = 28.0
    for hour in range(5):
        temp = model.step(temp, 35.0, 1, 1, 1.0)
        print(f"Hour {hour+1}: {temp:.2f}°C")
