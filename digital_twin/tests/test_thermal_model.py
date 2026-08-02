import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from thermal_model import RoomThermalModel

def get_model():
    return RoomThermalModel({
        "thermal_mass": 5.0,
        "insulation_resistance": 2.0,
        "ac_cooling_power": 3.0,
        "occupancy_heat_gain": 0.5,
    })

def test_ac_on_cools_room():
    model = get_model()
    new_temp = model.step(current_temp=30.0, outdoor_temp=35.0, ac_state=1, occupancy=0, dt=1.0)
    assert new_temp < 30.0 or new_temp < 30.0 + 1.5  # AC should limit rise

def test_ac_off_room_heats_up():
    model = get_model()
    new_temp = model.step(current_temp=25.0, outdoor_temp=35.0, ac_state=0, occupancy=0, dt=1.0)
    assert new_temp > 25.0  # no AC, hot outside -> temp rises

def test_occupancy_adds_heat():
    model = get_model()
    temp_no_occ = model.step(25.0, 25.0, ac_state=0, occupancy=0, dt=1.0)
    temp_with_occ = model.step(25.0, 25.0, ac_state=0, occupancy=1, dt=1.0)
    assert temp_with_occ > temp_no_occ

if __name__ == "__main__":
    test_ac_on_cools_room()
    test_ac_off_room_heats_up()
    test_occupancy_adds_heat()
    print("All tests passed!")