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
from util import pickle_save
from arm_distribution_generator import pareto
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
plt.ioff()

def is_pareto_efficient_simple(costs): #minimization
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def plot_pareto(pareto_opt_ind, dominated_ind, Y):
    fig = plt.figure()
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)

    ax.scatter(Y[pareto_opt_ind, 0], Y[pareto_opt_ind,1], color= 'red', alpha= 0.2)
    ax.scatter(Y[dominated_ind, 0], Y[dominated_ind,1], color= 'blue', alpha= 0.2)
    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel("X axis label")
    plt.ylabel("Y axis label")

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['xtick.major.pad'] = '5'
    plt.rcParams['ytick.major.pad'] = '5'

    plt.savefig('test.pdf', bbox_inches = 'tight', pad_inches = 1)



def get_true_bg_levels(food):
    # --------- Create Random Scenario --------------
    arms= np.linspace(2, 30, 29, dtype= int)
    K = 29

    after_90_list = []
    after_180_list= []

    for bolus_tot in arms:
        # specify start_time as the beginning of today
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        food= 35  #carb = food/3 * 3 = food
        # bolus_tot= 6
        basal = 0
        patient_name= 'adult#002'

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

    dist_90 = -np.abs(np.array(after_90_list)-140)
    dist_180= -np.abs(np.array(after_180_list)-140)

    Y = np.zeros([K, 2])
    Y[:, 0] = dist_90[:]
    Y[:, 1] = dist_180[:]
    pareto_opt_ind, dominated_ind= pareto(Y, K)
    plot_pareto(pareto_opt_ind, dominated_ind, -Y)



    x = np.zeros([K,1])
    x[:, 0]= arms
        # is_pareto_efficient_simple(-Y)
    diabetes_dict= dict()
    diabetes_dict['patient'] = patient_name
    diabetes_dict['food']= food
    diabetes_dict['y'] = Y
    diabetes_dict['x'] = x
    diabetes_dict['after_90_list'] = after_90_list
    diabetes_dict['after_180_list'] = after_180_list
    pickle_save('diabetes_dict.pickle', diabetes_dict)

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
