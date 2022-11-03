"""
Authors: Mateusz Pioch s21331, Stanisław Dominiak s18864

Based on http://www.arpnjournals.org/jeas/research_papers/rp_2018/jeas_0118_6698.pdf

Ensure you have the correct dependencies installed:
- Python 3.10
- skfuzzy (pip3 install scikit-fuzzy)
- Numpy (pip3 install numpy)
- matplotlib (pip3 install matplotlib)
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""

"""

"""
Ranges
"""
"""
input ranges
"""
x_humidity = np.arange(0, 100, 1)
x_temperature = np.arange(0, 100, 1)
x_ilumination = np.arange(0, 500, 1)

"""
output ranges
"""
x_lamp = np.arange(0, 500, 1)
x_water_pump = np.arange(0, 400, 1)

"""
setting up the fuzzy varibles
"""
humidity = ctrl.Antecedent(x_humidity, 'humidity')
temperature = ctrl.Antecedent(x_temperature, 'temperature')
ilumination = ctrl.Antecedent(x_ilumination, 'ilumination')

lamp = ctrl.Consequent(x_lamp, 'lamp')
water_pump = ctrl.Consequent(x_water_pump, 'water_pump')


"""

"""
humidity['dry'] = fuzz.trapmf(x_humidity, [0, 0, 35, 35])
humidity['normal'] = fuzz.trimf(x_humidity, [35, 50, 70])
humidity['moist'] = fuzz.trapmf(x_humidity, [70, 70, 100, 100])

temperature['low'] = fuzz.trapmf(x_temperature, [0, 0, 30, 30])
temperature['medium'] = fuzz.trimf(x_temperature, [30, 35, 40])
temperature['high'] = fuzz.trapmf(x_temperature, [40, 40, 100, 100])

ilumination['dark'] = fuzz.trapmf(x_ilumination, [0, 0, 160, 160])
ilumination['normal'] = fuzz.trimf(x_ilumination, [160, 250, 340])
ilumination['bright'] = fuzz.trapmf(x_ilumination, [340, 340, 500, 500])

lamp['low'] = fuzz.trapmf(x_lamp, [0, 0, 150, 150])
lamp['medium'] = fuzz.trimf(x_lamp, [150, 225, 300])
lamp['high'] = fuzz.trapmf(x_lamp, [300, 300, 500, 500])

water_pump['off'] = fuzz.trapmf(x_water_pump, [0, 0, 150, 150])
water_pump['on'] = fuzz.trapmf(x_water_pump, [150, 150, 400, 400])



# """
# Rules according to which output is calculated
# """

rules = [
    ctrl.Rule(humidity['dry'] & temperature['low'] & ilumination['dark'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['normal'] & temperature['low'] & ilumination['dark'], (lamp['high'], water_pump['off'])),
    ctrl.Rule(humidity['moist'] & temperature['low'] & ilumination['dark'], (lamp['high'], water_pump['off'])),

    ctrl.Rule(humidity['dry'] & temperature['low'] & ilumination['normal'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['normal'] & temperature['low'] & ilumination['normal'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['moist'] & temperature['low'] & ilumination['normal'], (lamp['high'], water_pump['off'])),

    ctrl.Rule(humidity['dry'] & temperature['low'] & ilumination['bright'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['normal'] & temperature['low'] & ilumination['bright'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['moist'] & temperature['low'] & ilumination['bright'], (lamp['medium'], water_pump['off'])),
    
    ctrl.Rule(humidity['dry'] & temperature['medium'] & ilumination['dark'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['normal'] & temperature['medium'] & ilumination['dark'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['moist'] & temperature['medium'] & ilumination['dark'], (lamp['high'], water_pump['off'])),

    ctrl.Rule(humidity['dry'] & temperature['medium'] & ilumination['normal'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['normal'] & temperature['medium'] & ilumination['normal'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['moist'] & temperature['medium'] & ilumination['normal'], (lamp['medium'], water_pump['off'])),

    ctrl.Rule(humidity['dry'] & temperature['medium'] & ilumination['bright'], (lamp['low'], water_pump['off'])),
    ctrl.Rule(humidity['normal'] & temperature['medium'] & ilumination['bright'], (lamp['medium'], water_pump['off'])),
    ctrl.Rule(humidity['moist'] & temperature['medium'] & ilumination['bright'], (lamp['medium'], water_pump['off'])),
    
    ctrl.Rule(humidity['dry'] & temperature['high'] & ilumination['dark'], (lamp['medium'], water_pump['on'])),
    ctrl.Rule(humidity['normal'] & temperature['high'] & ilumination['dark'], (lamp['medium'], water_pump['on'])),
    ctrl.Rule(humidity['moist'] & temperature['high'] & ilumination['dark'], (lamp['medium'], water_pump['on'])),

    ctrl.Rule(humidity['dry'] & temperature['high'] & ilumination['normal'], (lamp['low'], water_pump['on'])),
    ctrl.Rule(humidity['normal'] & temperature['high'] & ilumination['normal'], (lamp['medium'], water_pump['on'])),
    ctrl.Rule(humidity['moist'] & temperature['high'] & ilumination['normal'], (lamp['medium'], water_pump['on'])),

    ctrl.Rule(humidity['dry'] & temperature['high'] & ilumination['bright'], (lamp['low'], water_pump['on'])),
    ctrl.Rule(humidity['normal'] & temperature['high'] & ilumination['bright'], (lamp['low'], water_pump['on'])),
    ctrl.Rule(humidity['moist'] & temperature['high'] & ilumination['bright'], (lamp['medium'], water_pump['on']))
]

output_ctrl = ctrl.ControlSystem(rules)
output_simulation = ctrl.ControlSystemSimulation(output_ctrl)

"""
Example simulation
"""
output_simulation.input['humidity'] = 90
output_simulation.input['temperature'] = 10
output_simulation.input['ilumination'] = 400
output_simulation.compute()

print("Wynik lampa: ", output_simulation.output['lamp']," oraz pompa wody: ", output_simulation.output['water_pump'])