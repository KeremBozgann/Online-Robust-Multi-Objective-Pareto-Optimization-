from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.custom_insulin import custom_insulin
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime
import numpy as np



def get_true_bg_levels(food):
    # --------- Create Random Scenario --------------

    after_90_list = []
    after_180_list= []

    for bolus_tot in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        # specify start_time as the beginning of today
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        food= 35  #carb = food/3 * 3 = food
        # bolus_tot= 6
        basal = 0
        patient_name= 'adult#001'

        bolus= bolus_tot/3 #total = bolus * 3
        # --------- Create Custom Scenario --------------
        # Create a simulation environment
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        # custom scenario is a list of tuples (time, meal_size)

        # scen = [(0, 45), (12, 70), (16, 15), (18, 80), (23, 10)]
        scen = [(0, food)]
        scenario = CustomScenario(start_time=start_time, scenario=scen)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller

        path = f'./results/{patient_name}_carb_{food}_ins_{bolus_tot}'

        # controller = BBController()
        controller = custom_insulin(bolus, basal)

        # Put them together to create a simulation object
        s2 = SimObj(env, controller, timedelta(hours=5), animate=False, path=path)
        results2 = sim(s2)

        res_numpy= results2.to_numpy()

        after_90_list.append(res_numpy[30][0])
        after_180_list.append(res_numpy[60][0])
        dist_90 = np.abs(np.array(after_90_list)-140)
        dist_180= np.abs(np.array(after_180_list)-140)

        return after_90_list, after_180_list
'''
# --------- batch simulation --------------
# Re-initialize simulation objects
s2.reset()

# create a list of SimObj, and call batch_sim
s = [ s2]
results = batch_sim(s, parallel=True)
print(results)
'''
