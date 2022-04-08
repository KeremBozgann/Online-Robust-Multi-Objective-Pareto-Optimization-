from algorithm1_auer import auer_algorithm1
from robust_moo import robust_moo
from find_alpha_suboptimal_arms import find_alpha_suboptimal_and_pareto, find_alpha_suboptimal_and_pareto_llvm, \
                                                                find_alpha_suboptimal_and_pareto_diabetes, find_alpha_suboptimal_and_pareto_new_algo
from plot_end import *
import pickle
from scipy.stats import norm
import numpy as np
from unavoidable_bias_calculator import UnavoidableBias
from plot_end import plot_medians_in_objective_space
import pickle
from util import *
from find_alpha_suboptimal_arms import check_covering
from create_table import latex_table

# paremeters
save_arms= False
load_previous_arms= False
# setting = 'synthetic'
# setting = 'llvm'
setting = 'diabetes'
bound_type = 'subgaussian'
# latex_code_output_file_name = 'latex_code_synthetic.txt'
latex_code_output_file_name = f'latex_code_{setting}.txt'
# data_dump_pickle_name ='run_synthetic.pickle'
data_dump_pickle_name =f'run_{setting}.pickle'
# bound_type = 'gaussian'
# setting = 'llvm'
# setting = 'diabetes'

dist_name = 'Gaussian'
advers = 'oblivious'
# spread = 10

N = 10


if setting == 'llvm':
    file_name  = 'llvm_dict.pickle'
    K = 16
    M = 2
    # std= 0.38
    std= 0.2
    delta = 0.1
    alpha = 0.1
    eps_max = 0.4
    eps_min = 0.0
    leng_epsilon_list = 9



elif setting == 'synthetic':
    K = 10
    M = 2
    # K_opt= 4
    std = 0.1
    reward_max = 10
    reward_min = 0

    delta = 0.1
    alpha = 0.1
    eps_max = 0.4
    eps_min = 0.0
    leng_epsilon_list = 9

elif setting == 'diabetes':
    file_name = 'diabetes_dict.pickle'
    K = 11
    M = 2
    std = 0.1
    alpha = 1

    delta = 2
    eps_max = 0.4
    eps_min = 0.0
    leng_epsilon_list = 9


epsilon0= alpha
epsilon_list = np.linspace(eps_min, eps_max, leng_epsilon_list)
N_eps= len(epsilon_list)


total_sample_matrix_auer = np.zeros([N_eps, N])
total_sample_matrix_robust = np.zeros([N_eps, N])
correct_pred_matrix_auer = np.zeros([N_eps, N])
correct_pred_matrix_robust = np.zeros([N_eps, N])

ratio_of_opt_pred_to_tot_pred_matrix_auer= np.zeros([N_eps, N])
ratio_of_opt_pred_to_tot_pred_matrix_robust= np.zeros([N_eps, N])

pred_arms_violate_sc2_matrix_auer= np.zeros([N_eps, N])
pred_arms_violate_sc2_matrix_robust= np.zeros([N_eps, N])

if load_previous_arms:
    with open('arms.pickle', 'rb') as f:
        arms_list= pickle.load(f)
else:
    arms_list = []


if advers == 'prescient' or 'oblivious':
    if bound_type == 'gaussian':
        t_bar= 0.45
        t_bar0= 0.45
    elif bound_type == 'subgaussian':
        t_bar= 0.49
        t_bar0 = 0.49

elif advers== 'malicious':
    if bound_type == 'gaussian':
        t_bar= 0.45
        t_bar0= 0.45
    elif bound_type == 'subgaussian':
        t_bar= 0.49
        t_bar0 = 0.49

#for gaussian distirbutions:
if bound_type == 'gaussian':
    B= norm.ppf(3/4)/norm.pdf(norm.ppf(1/2+ t_bar))
    B0 = norm.ppf(3/4)/norm.pdf(norm.ppf(1/2+ t_bar0))
elif bound_type == 'subgaussian':
    pass

if bound_type == 'gaussian':
    if setting == 'llvm':
        m2 = norm.ppf(3/4)* std
    elif setting == 'synthetic':
        m2 = norm.ppf(3/4)
    elif setting == 'diabetes':
        m2 = norm.ppf(3/4)* std

    R = lambda t: t * B * m2
    R0 = lambda t: t * B0 * m2
elif bound_type=='subgaussian':
    R = lambda t: std*np.sqrt(2)*(np.sqrt(np.log(1/(1/2-t)))+np.sqrt(np.log(2)))
    # h_epsilon = lambda epsilon: epsilon / (2 * (1 - epsilon))
    # h_eps = h_epsilon(eps_max)

import importlib as imp
import arm_distribution_generator
imp.reload(arm_distribution_generator)
from arm_distribution_generator import *

if setting == 'llvm':

    arms= ArmGenerator(K, M, dist_name)
    arms.load_llvm(file_name)

elif setting == 'synthetic':
    arms_list= []
    for i in range(N):
        arms = ArmGenerator(K, M, dist_name)
        arms.create_medians_2obj_new_algo(reward_min, reward_max)
        arms_list.append(arms)

elif setting == 'diabetes':
    arms= ArmGenerator(K, M, dist_name)
    arms.load_diabetes(file_name)

if setting== 'synthetic':
    ind_suboptimal_matrix= np.empty([N, leng_epsilon_list])
    Pareto_ind_list= []
    non_pareto_ind_list  = []
    for i in range(N):

        Pareto_ind, non_pareto_ind = pareto(arms_list[i].median_matrix, K)
        Pareto_ind_list.append(Pareto_ind)
        non_pareto_ind_list.append(non_pareto_ind)



# plot_medians_in_objective_space(arms_list[i].median_matrix, lims= [[reward_min, reward_max], [reward_min, reward_max]])

ind_suboptimal_list_of_lists= list()
# z= 0 ; epsilon = 0.4
for z, epsilon in enumerate(epsilon_list):
    print('epsilon', epsilon)
    bias = UnavoidableBias(R, epsilon, advers)
    D = bias.return_D()

    # plot_reward_medians_new_algo(arms, K, D, False)

    if setting=='synthetic':
        ind_suboptimal_list= list()
        for k in range(N):
            ind_suboptimal = find_alpha_suboptimal_and_pareto_new_algo(arms_list[k].median_matrix, D, K, alpha)
            ind_suboptimal_list.append(ind_suboptimal)
        ind_suboptimal_list_of_lists.append(ind_suboptimal_list)
    elif setting == 'llvm':
        ind_suboptimal, Pareto_ind, non_pareto_ind = find_alpha_suboptimal_and_pareto_llvm(arms.y, arms.sample_inds_dict, D, K, alpha)
    elif setting == 'diabetes':
        ind_suboptimal, Pareto_ind, non_pareto_ind = find_alpha_suboptimal_and_pareto_diabetes(arms.y,
                                                                                               D, K,
                                                                                               epsilon0)

    total_samples_auer = np.zeros([N, ])
    total_samples_robust = np.zeros([N, ])
    correct_pred_auer = np.zeros([N, ])
    correct_pred_robust = np.zeros([N, ])

    ratio_of_opt_pred_to_tot_pred_auer= np.zeros([N, ])
    ratio_of_opt_pred_to_tot_pred_robust= np.zeros([N, ])
    pred_arms_violate_sc2_auer= np.zeros([N, ])
    pred_arms_violate_sc2_robust= np.zeros([N, ])

    for i in range(N):
        print('iter', i+ 1)
        if setting == 'synthetic':
            arms= arms_list[i]
            ind_suboptimal= ind_suboptimal_list[i]
            Pareto_ind= Pareto_ind_list[i]
            non_pareto_ind= non_pareto_ind_list[i]

        P_auer, eliminated_auer, P_auer_ind= auer_algorithm1(K, M, epsilon, delta, epsilon0, arms, setting, Pareto_ind, std, D)
        if bound_type == 'gaussian':
            if epsilon <= t_bar0:
                P_robust, eliminated_robust, P_robust_ind= robust_moo(advers, t_bar0, K, M, R0, epsilon, alpha, delta, arms, setting, Pareto_ind, std, D)

            else:
                P_robust, eliminated_robust, P_robust_ind = robust_moo(advers, t_bar, K, M, R, epsilon, alpha, delta, arms,
                                                                   setting, Pareto_ind, D)

        elif bound_type == 'subgaussian':
            P_robust, eliminated_robust, P_robust_ind = robust_moo(advers, t_bar, K, M, R, epsilon, alpha, delta,
                                                                   arms, setting, Pareto_ind, std, D)

        suboptimal_auer= True
        for k in P_auer_ind:
            if not k in ind_suboptimal:
                suboptimal_auer= False
        covering_auer= True
        covering_auer= check_covering(arms.median_matrix, D, alpha , P_auer_ind, Pareto_ind)
        # for j in Pareto_ind:
        #     if not j in P_auer_ind:
        #         includes_pareto_auer= False
        if suboptimal_auer and covering_auer:
            correct_pred_auer[i]= True


        suboptimal_robust= True
        for k in P_robust_ind:
            if not k in ind_suboptimal:
                suboptimal_robust= False
        covering_robust= True
        covering_robust= check_covering(arms.median_matrix, D, alpha , P_robust_ind, Pareto_ind)
        # for j in Pareto_ind:
        #     if not j in P_robust_ind:
        #         includes_pareto_robust= False

        if suboptimal_robust and covering_robust:
            correct_pred_robust[i]= True

        auer_samp= 0
        auer_dict= {**P_auer, **eliminated_auer}
        for arm in auer_dict:
            auer_samp+= auer_dict[arm]['ti']

        robust_samp= 0
        robust_dict= {**P_robust, **eliminated_robust}
        for arm in auer_dict:
            robust_samp+= robust_dict[arm]['Ni']
        total_samples_auer[i]= auer_samp
        total_samples_robust[i]= robust_samp



        num_pareto_in_pred_auer= 0
        for ind in P_auer_ind:
            if ind in Pareto_ind:
                num_pareto_in_pred_auer+= 1
        ratio_of_opt_pred_to_tot_pred_auer[i] = num_pareto_in_pred_auer/len(Pareto_ind)

        num_pareto_in_pred_robust= 0
        for ind in P_robust_ind:
            if ind in Pareto_ind:
                num_pareto_in_pred_robust+= 1
        ratio_of_opt_pred_to_tot_pred_robust[i] = num_pareto_in_pred_robust/len(Pareto_ind)



        num_violate_sc2_auer= 0
        for ind in P_auer_ind:
            if ind in non_pareto_ind:
                if ind not in ind_suboptimal:
                    num_violate_sc2_auer+=1
        pred_arms_violate_sc2_auer[i] = num_violate_sc2_auer

        num_violate_sc2_robust= 0
        for ind in P_robust_ind:
            if ind in non_pareto_ind:
                if ind not in ind_suboptimal:
                    num_violate_sc2_robust+=1
        pred_arms_violate_sc2_robust[i] = num_violate_sc2_robust


    total_sample_matrix_auer[z, :] = total_samples_auer[:]
    total_sample_matrix_robust[z, :] = total_samples_robust[:]
    correct_pred_matrix_auer[z, :] = correct_pred_auer[:]
    correct_pred_matrix_robust[z, :] = correct_pred_robust[:]

    ratio_of_opt_pred_to_tot_pred_matrix_auer[z, :]= ratio_of_opt_pred_to_tot_pred_auer[:]
    ratio_of_opt_pred_to_tot_pred_matrix_robust[z, :]= ratio_of_opt_pred_to_tot_pred_robust[:]
    pred_arms_violate_sc2_matrix_auer[z, :]  = pred_arms_violate_sc2_auer[:]
    pred_arms_violate_sc2_matrix_robust[z, :]  = pred_arms_violate_sc2_robust[:]

    print('auer correct pred', np.mean(correct_pred_auer), 'auer total samp', np.mean(total_samples_auer))
    print('correct pred robust', np.mean(correct_pred_robust), 'robuts total samp;', np.mean(total_samples_robust))

if save_arms:
    with open('arms.pickle', 'wb') as f:
        pickle.dump(arms_list, f)


total_samp_mean_auer= np.mean(total_sample_matrix_auer, axis = 1)
total_samp_mean_robust= np.mean(total_sample_matrix_robust, axis= 1)
total_samp_std_auer= np.std(total_sample_matrix_auer, axis= 1)
total_samp_std_robust= np.std(total_sample_matrix_robust, axis= 1)

correct_pred_mean_auer= np.mean(correct_pred_matrix_auer, axis= 1)
correct_pred_std_auer= np.std(correct_pred_matrix_robust, axis= 1)
correct_pred_mean_robust= np.mean(correct_pred_matrix_robust, axis= 1)
correct_pred_std_robust= np.std(correct_pred_matrix_robust, axis= 1)

ratio_of_opt_pred_to_tot_pred_mean_auer= np.mean(ratio_of_opt_pred_to_tot_pred_matrix_auer, axis= 1)
ratio_of_opt_pred_to_tot_pred_mean_robust= np.mean(ratio_of_opt_pred_to_tot_pred_matrix_robust, axis= 1)
pred_arms_violate_sc2_mean_auer= np.mean(pred_arms_violate_sc2_matrix_auer, axis= 1)
pred_arms_violate_sc2_mean_robust= np.mean(pred_arms_violate_sc2_matrix_robust, axis= 1)

print('samp auer:', total_samp_mean_auer)
print('samp robust:', total_samp_mean_robust)

print('correct pred auer:', correct_pred_mean_auer)
print('correct pred robust:', correct_pred_mean_robust)

results_dict= dict()
results_dict['auer'] = dict()
results_dict['auer']['total_sample_matrix_auer'] = total_sample_matrix_auer
results_dict['auer']['total_samp_mean_auer'] = total_samp_mean_auer
results_dict['auer']['correct_pred_matrix_auer'] = correct_pred_matrix_auer
results_dict['auer']['correct_pred_mean_auer'] = correct_pred_mean_auer
results_dict['auer']['ratio_of_opt_pred_to_tot_pred_matrix_auer'] = ratio_of_opt_pred_to_tot_pred_matrix_auer
results_dict['auer']['ratio_of_opt_pred_to_tot_pred_mean_auer'] = ratio_of_opt_pred_to_tot_pred_mean_auer
results_dict['auer']['pred_arms_violate_sc2_matrix_auer'] = pred_arms_violate_sc2_matrix_auer
results_dict['auer']['pred_arms_violate_sc2_mean_auer'] = pred_arms_violate_sc2_mean_auer

results_dict['robust'] = dict()
results_dict['robust']['total_sample_matrix_robust'] = total_sample_matrix_robust
results_dict['robust']['total_samp_mean_robust'] = total_samp_mean_robust
results_dict['robust']['correct_pred_matrix_robust'] = correct_pred_matrix_robust
results_dict['robust']['correct_pred_mean_robust'] = correct_pred_mean_robust
results_dict['robust']['ratio_of_opt_pred_to_tot_pred_matrix_robust'] = ratio_of_opt_pred_to_tot_pred_matrix_robust
results_dict['robust']['ratio_of_opt_pred_to_tot_pred_mean_robust'] = ratio_of_opt_pred_to_tot_pred_mean_robust
results_dict['robust']['pred_arms_violate_sc2_matrix_robust'] = pred_arms_violate_sc2_matrix_robust
results_dict['robust']['pred_arms_violate_sc2_mean_robust'] = pred_arms_violate_sc2_mean_robust


pickle_save(data_dump_pickle_name, results_dict)

latex_table(epsilon_list, leng_epsilon_list, latex_code_output_file_name,
            correct_pred_mean_auer, correct_pred_mean_robust,
            total_samp_mean_auer, total_samp_mean_robust,
            ratio_of_opt_pred_to_tot_pred_mean_auer,  ratio_of_opt_pred_to_tot_pred_mean_robust,
            pred_arms_violate_sc2_mean_auer, pred_arms_violate_sc2_mean_robust)
plot_pred_ratios(epsilon_list, correct_pred_mean_auer, correct_pred_std_auer, correct_pred_mean_robust, correct_pred_std_robust)
plot_sample_nums(epsilon_list, total_samp_mean_auer, total_samp_std_auer, total_samp_mean_robust, total_samp_std_robust)


