import numpy as np
from sample import Arm
from arm_distribution_generator import ArmGenerator
from scipy.stats import norm
from maximum_uncertainty_arm import find_arm_with_maximum_uncertainty
# from arm_distribution_generator import return_pessimistic_pareto_adversarial_index
from arm_distribution_generator import find_pessimistic_and_eliminate
from get_O1_set import identify_optimal_arms
from get_O2_set import identify_O2_set
from arm_distribution_generator import plot_arms_median
from arm_distribution_generator import pessimistic_pareto_adversarial, plot_arms_empirical
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from arm_distribution_generator import plot_arm_empirical_with_true_median, pareto
from find_alpha_suboptimal_arms import find_alpha_suboptimal_and_pareto, find_alpha_suboptimal_and_pareto_llvm, \
                                        find_alpha_suboptimal_and_pareto_diabetes,find_alpha_suboptimal_and_pareto_new_algo


def robust_moo(advers, t_bar, K, M, R, epsilon, alpha, delta, arms, setting, Pareto_ind, std, D):
    if advers == 'prescient':
        eps_bound = (2 * t_bar) / (1 + 2 * t_bar)
    elif advers == 'oblivious':
        eps_bound = t_bar

    arm_dict = dict()
    debug= False
    N_theoretical= True
    U_theoretical= True
    for i in range(K):
        arm_dict[str(i)] = dict()
        arm_dict[str(i)]['ti'] = 0
        arm_dict[str(i)]['Ui'] = 0
        arm_dict[str(i)]['Ni']= 0
        arm_dict[str(i)]['mi_hat'] = np.zeros([M, ])
        arm_dict[str(i)]['samples']  = dict()

        for obj in range(M):
            arm_dict[str(i)]['samples'][obj]  = np.zeros([0, ])

    #begin
    if advers == 'prescient' or 'oblivious':
        beta = np.power(t_bar - epsilon/(2*(1-epsilon)), -2)
        h_eps= epsilon/(2* (1-epsilon))
        delta_tild = delta/2
    elif advers == 'malicious':
        beta = np.power((t_bar - epsilon), -2)
        h_eps= epsilon
        delta_tild=delta/3
    else:
        print('incorrect adversary name')


    #init class sample
    # dist_name = 'Gaussian'
    # mean, var= 0.0, 1.0

    if N_theoretical:
        N_init = np.ceil(2 * beta * np.log((np.pi**2*M*K)/(6 * delta_tild)))
    else :
        _N_init = np.ceil(2 * beta * np.log((np.pi ** 2 * M * K) / (6 * delta_tild)))
        if _N_init < 100:
            N_init= _N_init
        else:
            N_init = np.ceil(_N_init/10)

    t = 0
    # plot_arms_median(arms.median_matrix, arms.D_matrix, M)
    for arm in arm_dict:
        arm_dict[arm]['ti'] +=1
        ti = arm_dict[arm]['ti']
        arm_dict[arm]['Ni'] += N_init
        if U_theoretical:
            arm_dict[arm]['Ui'] = R(h_eps + 1/np.sqrt(beta* ti)) - R(h_eps)
        else:
            # arm_dict[arm]['Ui'] = (R(h_eps + (t_bar - epsilon) / (np.sqrt(ti))) - R(h_eps))/10
            # arm_dict[arm]['Ui'] = R(h_eps + 1 / (np.sqrt(beta) * ti)) - R(h_eps)
            eps_max= 0.4
            beta0 = np.power(t_bar - eps_max / (2 * (1 - eps_max)), -2)
            arm_dict[arm]['Ui'] = R(h_eps + 1 / np.sqrt(beta0 * ti)) - R(h_eps)

        for i in range(M):
            if setting == 'synthetic':
                arm_dict[arm]['samples'][i]= \
                    np.append(arm_dict[arm]['samples'][i], arms.create_samples3(int(arm), i, int(N_init), epsilon, Pareto_ind, std), axis= 0)
                arm_dict[arm]['mi_hat'][i] = np.median( arm_dict[arm]['samples'][i])
            elif setting == 'llvm':
                arm_dict[arm]['samples'][i]= \
                    np.append(arm_dict[arm]['samples'][i], arms.create_samples_llvm2(int(arm), i, int(N_init), epsilon, Pareto_ind), axis= 0)
                arm_dict[arm]['mi_hat'][i] = np.median( arm_dict[arm]['samples'][i])
            elif setting == 'diabetes':
                arm_dict[arm]['samples'][i]= \
                    np.append(arm_dict[arm]['samples'][i], arms.create_samples_diabetes(int(arm), i, int(N_init), epsilon, std), axis= 0)
                arm_dict[arm]['mi_hat'][i] = np.median( arm_dict[arm]['samples'][i])

        if debug:
            for arm in arm_dict:
                fig, ax = plt.subplots(1)
                m_obj1 = np.median(arm_dict[arm]['samples'][0])
                m_obj2 = np.median(arm_dict[arm]['samples'][1])
                ax.scatter(m_obj1, m_obj2, color='red')
                ax.scatter(arms.median_matrix[int(arm), 0], arms.median_matrix[int(arm), 1])
                rect = mpatches.Rectangle((arms.median_matrix[int(arm), 0] - arm_dict[arm]['Ui'],
                                           arms.median_matrix[int(arm), 1] - arm_dict[arm]['Ui']),
                                          2 * arm_dict[arm]['Ui'],
                                          2 * arm_dict[arm]['Ui'],
                                          fill=False,
                                          color="purple",
                                          linewidth=1)
                ax.annotate(arm, (arms.median_matrix[int(arm), 0], arms.median_matrix[int(arm), 1]))
                plt.gca().add_patch(rect)
    #find the arm with maximum uncertainty
    P= dict()
    eliminated= dict()
    while True:
        if t> 0:
            _, arm_max= find_arm_with_maximum_uncertainty(arm_dict)
            arm_dict[arm_max]['ti'] += 1
            ti = arm_dict[arm_max]['ti']
            if N_theoretical:
                delt_Ni = np.ceil(1 + 4* ti * beta * np.log(ti/ (ti-1)) + 2 * beta * np.log((ti - 1 ) ** 2 * M * K * np.pi ** 2 / (6 * delta_tild)))
            else:
                _delt_Ni= np.ceil(1 + 4* ti * beta * np.log(ti/ (ti-1)) + 2 * beta * np.log((ti - 1 ) ** 2 * M * K * np.pi ** 2 / (6 * delta_tild)))
                if _delt_Ni < 100:
                    delt_Ni = _delt_Ni
                else:
                    delt_Ni = np.ceil(_delt_Ni / 10)

            arm_dict[arm_max]['Ni'] += delt_Ni
            if U_theoretical:
                arm_dict[arm_max]['Ui'] = R(h_eps + 1/np.sqrt(beta* ti)) - R(h_eps)
            else:
                # arm_dict[arm_max]['Ui'] = (R(h_eps + (t_bar- epsilon)/(np.sqrt(ti))) - R(h_eps))/10
                # arm_dict[arm_max]['Ui'] =  R(h_eps + 1/ (np.sqrt(beta) * ti) )- R(h_eps)
                eps_max= 0.4
                beta0 = np.power(t_bar - eps_max / (2 * (1 - eps_max)), -2)
                arm_dict[arm_max]['Ui'] = R(h_eps + 1 / np.sqrt(beta0 * ti)) - R(h_eps)

            for i in range(M):
                if setting == 'synthetic':
                    arm_dict[arm_max]['samples'][i]= np.append(arm_dict[arm_max]['samples'][i],
                                                                arms.create_samples3(int(arm_max), i, int(delt_Ni), epsilon, Pareto_ind, std), axis= 0)
                elif setting == 'llvm':
                    arm_dict[arm_max]['samples'][i]= np.append(arm_dict[arm_max]['samples'][i],
                                                                arms.create_samples_llvm2(int(arm_max), i, int(delt_Ni), epsilon, Pareto_ind), axis= 0)
                elif setting == 'diabetes':
                    arm_dict[arm_max]['samples'][i]= np.append(arm_dict[arm_max]['samples'][i],
                                                                arms.create_samples_diabetes(int(arm_max), i, int(delt_Ni), epsilon, std), axis= 0)

                arm_dict[arm_max]['mi_hat'][i] = np.median( arm_dict[arm_max]['samples'][i])
        if debug:
            fig, ax = plt.subplots(1)
            m_obj1 = np.median(arm_dict[arm_max]['samples'][0])
            m_obj2 = np.median(arm_dict[arm_max]['samples'][1])
            ax.scatter(m_obj1, m_obj2, color='red')
            ax.scatter(arms.median_matrix[int(arm_max), 0], arms.median_matrix[int(arm_max), 1])
            rect = mpatches.Rectangle((arms.median_matrix[int(arm_max), 0] - arm_dict[arm_max]['Ui'],
                                       arms.median_matrix[int(arm_max), 1] - arm_dict[arm_max]['Ui']),
                                      2 * arm_dict[arm_max]['Ui'],
                                      2 * arm_dict[arm_max]['Ui'],
                                      fill=False,
                                      color="purple",
                                      linewidth=1)
            plt.gca().add_patch(rect)
            rect = mpatches.Rectangle((arms.median_matrix[int(arm_max), 0] - arms.D_matrix[int(arm_max), 0],
                                       arms.median_matrix[int(arm_max), 1] - arms.D_matrix[int(arm_max), 1]),
                                      2 * arms.D_matrix[int(arm_max), 0],
                                      2 * arms.D_matrix[int(arm_max), 1],
                                      fill=False,
                                      color="orange",
                                      linewidth=1)

            ax.annotate(arm_max, (arms.median_matrix[int(arm_max), 0], arms.median_matrix[int(arm_max), 1]))
            plt.gca().add_patch(rect)
            plot_arm_empirical_with_true_median(arm_dict, P, M, arms)
            # plot_arms_empirical(arm_dict, P, M)

        # pess_ind= return_pessimistic_pareto_adversarial_index(arm_dict, arms.D_matrix, M, K)

        ind_eliminated= find_pessimistic_and_eliminate(arm_dict, M, D)

        if ind_eliminated is None:
            pass
        else:
            # print('eliminated ind', ind_eliminated)
            for ind in ind_eliminated:
                eliminated[str(int(ind))]= arm_dict.pop(str(int(ind)))

        if len(arm_dict.keys()) == 0:
            break


        O1_index, U_vec= identify_optimal_arms(arm_dict, alpha, M)

        if np.any(U_vec > alpha/4):
            if not O1_index is None:
                # print('Ind O1', O1_index)
                O2_ind= identify_O2_set(O1_index, arm_dict, alpha, M)
                if O2_ind is not None:
                    # print('Ind O2', O2_ind)
                    for j in O2_ind:

                        P[str(int(j))]= arm_dict.pop(str(int(j)))

                    if len(arm_dict.keys()) == 0:
                        break
        else:
            if not O1_index is None:
                for j in O1_index:
                    P[str(int(j))] = arm_dict.pop(str(int(j)))

            arm_index = list()
            for arm in arm_dict:
                arm_index.append(int(arm))

            for arm_ind in arm_index:
                eliminated[str(int(arm_ind))] = arm_dict.pop(str(int(arm_ind)))
            break

        if debug:
            fig, ax = plt.subplots(1)
            m_obj1 = np.median(arm_dict[arm_max]['samples'][0])
            m_obj2 = np.median(arm_dict[arm_max]['samples'][1])
            ax.scatter(m_obj1, m_obj2, color='red')
            ax.scatter(arms.median_matrix[int(arm_max), 0], arms.median_matrix[int(arm_max), 1])
            rect = mpatches.Rectangle((arms.median_matrix[int(arm_max), 0] - arm_dict[arm_max]['Ui'],
                                       arms.median_matrix[int(arm_max), 1] - arm_dict[arm_max]['Ui']),
                                      2 * arm_dict[arm_max]['Ui'],
                                      2 * arm_dict[arm_max]['Ui'],
                                      fill=False,
                                      color="purple",
                                      linewidth=1)
            ax.annotate(arm_max, (arms.median_matrix[int(arm_max), 0], arms.median_matrix[int(arm_max), 1]))
            plt.gca().add_patch(rect)
            plot_arm_empirical_with_true_median({**arm_dict, **eliminated}, P, M, arms)

            for arm in arm_dict:
                print(arm_dict[arm]['ti'])
        t += 1
        # plot_arms_median(empirical_median_matrix, np.repeat(np.expand_dims(U_vec, axis= 1),M, 1),M)
    # plot_arms_median(arms.median_matrix, arms.D_matrix, M)
    # plot_arms_empirical(eliminated, P, M)
    # empirical_median_matrix = np.zeros([0, M])
    # U_vec = np.zeros([0, ])
    # for arm in eliminated:
    #     empirical_median_vec = np.zeros([1, M])
    #     empirical_median_vec[0, :] = arm_dict[arm]['mi_hat']
    #     empirical_median_matrix = np.append(empirical_median_matrix, empirical_median_vec, axis=0)
    #     U = arm_dict[arm]['Ui']
    #     U_vec = np.append(U_vec, np.array([U]), axis=0)

    if setting == 'synthetic':
        ind_suboptimal = find_alpha_suboptimal_and_pareto_new_algo(arms.median_matrix, D, K, alpha)
        P_star_ind, _ = pareto(arms.median_matrix, K)
    elif setting =='llvm':
        ind_suboptimal, P_star_ind, non_pareto_ind = find_alpha_suboptimal_and_pareto_llvm(arms.y,
                                                                                           arms.sample_inds_dict,
                                                                                           D, K, alpha)

    elif setting == 'diabetes':
        ind_suboptimal, P_star_ind, non_pareto_ind = find_alpha_suboptimal_and_pareto_diabetes(arms.y,
                                                                                               D, K, alpha)

    print('true P arms', P_star_ind, ' predicted P arms', P.keys(),
          'P star and alpha suboptimal arms', ind_suboptimal)
    P_ind= np.zeros([0, ])
    for arm in P:
        P_ind= np.append(P_ind, np.array([int(arm)]), axis= 0)
    return P, eliminated, P_ind

def  test_rmoo():
    # paremeters
    advers = 'prescient'
    alpha = 0.1
    dist_name = 'Gaussian'
    delta = 0.1
    K = 10
    M = 3
    t_bar = 0.2

    R = lambda t: t * 2.63
    epsilon = 0.00
    arms= ArmGenerator(K, M, dist_name, epsilon, advers, R)
    arms.create_medians()

    P, eliminated= robust_moo(advers, t_bar, K, M, R, epsilon, alpha, delta, arms)

