import numpy as np
from unavoidable_bias_calculator import UnavoidableBias
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.patches as mpatches
from util import pickle_read, pickle_save

def pessimistic_pareto_adversarial(D_matrix, median_matrix, K):
    lower_confidence= median_matrix- D_matrix
    non_dominated= np.ones([K])
    dominated= np.zeros([K])
    for i in range(K):
        for j in range(i+1, K):
            if np.all(lower_confidence[i] <= lower_confidence[j]):
                non_dominated[i] = 0
                dominated[i] = 1
            if np.all(lower_confidence[i] >= lower_confidence[j]):
                non_dominated[j] = 0
                dominated[j] = 1
    return non_dominated.nonzero()[0], dominated.nonzero()[0]

def test_pessimistic_pareto_adversarial():
    K, M= 10, 2
    D_matrix= np.random.rand(K,M)
    median_matrix= np.random.rand(K, M)
    print(median_matrix- D_matrix)
    print(pessimistic_pareto_adversarial(D_matrix, median_matrix, K))

def pareto(matrix, K):
    non_dominated= np.ones([K])
    dominated= np.zeros([K])
    for i in range(K):
        for j in range(i+1, K):
            if np.all(matrix[i] <= matrix[j]):
                non_dominated[i] =  0
                dominated[i] = 1
            elif np.all(matrix[j] <= matrix[i]):
                non_dominated[j] = 0
                dominated[j]= 1
    return non_dominated.nonzero()[0], dominated.nonzero()[0]

def plot_arms_median(median_matrix, D_matrix, M):
    if M == 2:
        fig, ax = plt.subplots()
        ax.scatter(median_matrix[:, 0] , median_matrix[:, 1])
        for i in range(median_matrix.shape[0]):
            rect = mpatches.Rectangle((median_matrix[i, 0]- D_matrix[i, 0], median_matrix[i,  1]- D_matrix[i,  1]),
                                      2 * D_matrix[i, 0] ,
                                      2 * D_matrix[i, 1],
                                      fill=False,
                                      color="purple",
                                      linewidth=1)
            ax.annotate(str(i), (median_matrix[i, 0], median_matrix[i, 1]))
            plt.gca().add_patch(rect)
        plt.show()

    else:
        print('number of objectives is not equal to 2')

def plot_arm_empirical_with_true_median(eliminated, P, M, arm_generator):
    index_list= np.zeros([0, ])
    median_matrix= np.zeros([0, M])
    U_vec= np.zeros([0, ])
    if M==2:
        for arm in eliminated:
            median_vec= np.expand_dims(eliminated[arm]['mi_hat'], axis=0)
            median_matrix= np.append(median_matrix, median_vec, axis = 0)
            index_list= np.append(index_list, np.array([int(arm)]), axis= 0)
            U= eliminated[arm]['Ui']
            U_vec= np.append(U_vec, np.array([U]), axis= 0)

        for arm in P:
            median_vec= np.expand_dims(P[arm]['mi_hat'], axis  = 0)
            median_matrix= np.append(median_matrix, median_vec, axis= 0)
            index_list = np.append(index_list, np.array([int(arm)]), axis= 0)
            U = P[arm]['Ui']
            U_vec= np.append(U_vec, np.array([U]), axis= 0)
        U_matrix= np.repeat(np.expand_dims(U_vec, axis=1), M, axis= 1)

        fig,ax= plt.subplots(1)
        ax.scatter(median_matrix[:, 0], median_matrix[:, 1], color= 'red')
        ax.scatter(arm_generator.median_matrix[:, 0], arm_generator.median_matrix[:, 1], color= 'blue')
        for i in range((arm_generator.median_matrix).shape[0]):
            rect = mpatches.Rectangle((arm_generator.median_matrix[i, 0] - arm_generator.D_matrix[i, 0] -U_matrix[np.where(index_list== i)[0][0], 0],
                                       arm_generator.median_matrix[i, 1] -arm_generator.D_matrix[i, 1] - U_matrix[np.where(index_list== i)[0][0], 1]),
                                      2 * ((arm_generator.D_matrix)[i, 0]+ U_matrix[np.where(index_list== i)[0][0], 0]),
                                      2 * ((arm_generator.D_matrix)[i, 1] + U_matrix[np.where(index_list== i)[0][0], 1]),
                                      fill=False,
                                      color="purple",
                                      linewidth=1)
            plt.gca().add_patch(rect)
            rect = mpatches.Rectangle((arm_generator.median_matrix[i, 0] - arm_generator.D_matrix[i, 0],
                                       arm_generator.median_matrix[i, 1] - arm_generator.D_matrix[i, 1]),
                                      2 * arm_generator.D_matrix[i, 0],
                                      2 * arm_generator.D_matrix[i, 1],
                                      fill=False,
                                      color="orange",
                                      linewidth=1)
            ax.annotate(str(index_list[i]), (median_matrix[i, 0], median_matrix[i, 1]))
            ax.annotate(str(i)+ ',t', (arm_generator.median_matrix[i, 0], arm_generator.median_matrix[i, 1]))
            plt.gca().add_patch(rect)
        plt.show()
    else:
        print('number of objectives is not equal to 2')



def plot_arms_empirical(eliminated, P, M):
    index_list= np.zeros([0, ])
    median_matrix= np.zeros([0, M])
    U_vec= np.zeros([0, ])
    if M==2:
        for arm in eliminated:
            median_vec= np.expand_dims(eliminated[arm]['mi_hat'], axis=0)
            median_matrix= np.append(median_matrix, median_vec, axis = 0)
            index_list= np.append(index_list, np.array([int(arm)]), axis= 0)
            U= eliminated[arm]['Ui']
            U_vec= np.append(U_vec, np.array([U]), axis= 0)

        for arm in P:
            median_vec= np.expand_dims(P[arm]['mi_hat'], axis  = 0)
            median_matrix= np.append(median_matrix, median_vec, axis= 0)
            index_list = np.append(index_list, np.array([int(arm)]), axis= 0)
            U = P[arm]['Ui']
            U_vec= np.append(U_vec, np.array([U]), axis= 0)
        U_matrix= np.repeat(np.expand_dims(U_vec, axis=1), M, axis= 1)

        fig,ax= plt.subplots(1)
        ax.scatter(median_matrix[:, 0], median_matrix[:, 1])
        for i in range(median_matrix.shape[0]):
            rect = mpatches.Rectangle((median_matrix[i, 0] - U_matrix[i, 0], median_matrix[i, 1] - U_matrix[i, 1]),
                                      2 * U_matrix[i, 0],
                                      2 * U_matrix[i, 1],
                                      fill=False,
                                      color="purple",
                                      linewidth=1)
            ax.annotate(str(index_list[i]), (median_matrix[i, 0], median_matrix[i, 1]))
            plt.gca().add_patch(rect)
        plt.show()
    else:
        print('number of objectives is not equal to 2')


def check_condition1(median_matrix, D_matrix, M, K):
    pess_ind, no_pess_ind = pessimistic_pareto_adversarial(D_matrix, median_matrix, K)
    for j in no_pess_ind:
        satisfied = False
        for i in pess_ind:
            if np.all((median_matrix[j]+ D_matrix[j]) <= (median_matrix[i] - D_matrix[i])):
                satisfied = True
                break
        # print(satisfied)
        if satisfied == False:
            return False
    return True

def check_condition2(median_matrix, D_matrix, M, K):
    pess_ind, no_pess_ind = pessimistic_pareto_adversarial(D_matrix, median_matrix, K)
    for i in range(pess_ind.shape[0]):
        for j in range(i+1, pess_ind.shape[0]):
            if np.all((median_matrix[pess_ind[i]] - D_matrix[pess_ind[i]])  <=
                      (median_matrix[pess_ind[j]] + D_matrix[pess_ind[j]])):
                return False
            if np.all((median_matrix[pess_ind[j]] - D_matrix[pess_ind[j]]) <=
                      (median_matrix[pess_ind[i]] + D_matrix[pess_ind[i]])):
                return False
    return True



def add_gap_to_satisfy_condition1(median_matrix, D_matrix, M, K):
    pess_index, non_pess_index = pessimistic_pareto_adversarial(D_matrix, median_matrix, K)
    for j in non_pess_index:
        satisfied = False
        gaps = np.zeros([pess_index.shape[0], M])
        for i, ind in enumerate(pess_index):
            gap = median_matrix[ind] - D_matrix[ind] - (median_matrix[j] + D_matrix[j])
            gaps[i, :] = gap
            if np.all(gap >= 0):
                satisfied = True
                break
        if not satisfied:
            dominate_opt_point_index = np.argmin(np.max(np.abs(np.clip(gaps, a_max=0, a_min=-np.inf)), axis=1))
            min_domination_gap = np.min(np.max(np.abs(np.clip(gaps, a_max=0, a_min=-np.inf)), axis=1))
            median_matrix[pess_index[dominate_opt_point_index]] += 2 * min_domination_gap
    return median_matrix,  D_matrix

def satisfy_condition_2(median_matrix, D_matrix, M, K):
    pess_index, non_pess_index = pessimistic_pareto_adversarial(D_matrix, median_matrix, K)

    for i in range(pess_index.shape[0]):
        ind1 = pess_index[i]
        for j in range(i+1, pess_index.shape[0]):
            ind2= pess_index[j]

            #check gap1
            gap1= (median_matrix[ind1]- D_matrix[ind1]) - (median_matrix[ind2] + D_matrix[ind2])
            if np.all(gap1 <= 0 ):
                delt_min= np.min(np.abs(gap1))
                dmin= np.argmin(np.abs(gap1))
                median_matrix[ind1][dmin] += 2 * delt_min
            #check gap2
            gap2 = (median_matrix[ind2] - D_matrix[ind2]) - (median_matrix[ind1] + D_matrix[ind1])
            if np.all(gap2 <= 0 ):
                delt_min= np.min(np.abs(gap2))
                dmin= np.argmin(np.abs(gap1))
                median_matrix[ind1][dmin] += 2 * delt_min
    return median_matrix, D_matrix

def create_arms_test(K, M):
    #create suboptimal
    while True:
        median_matrix= np.random.rand(K, M)
        D_matrix= np.random.rand(K, M)/10
        pess_index, non_pess_index= pessimistic_pareto_adversarial(D_matrix, median_matrix, K)
        print('pess_index:', pess_index, 'non_pess index:', non_pess_index)
        plot_arms_median(median_matrix, D_matrix, M)
        print('condition 1:',  check_condition1(median_matrix, D_matrix, M, K))
        print('condition 2:' ,check_condition2(median_matrix, D_matrix, M, K))
        while True:
            median_matrix, D_matrix = add_gap_to_satisfy_condition1(median_matrix, D_matrix, M)
            # plot_arms_median(median_matrix, D_matrix, M)
            if check_condition1(median_matrix, D_matrix, M, K):
                break
        print('condition 1:',  check_condition1(median_matrix, D_matrix, M, K))
        print('condition 2:' ,check_condition2(median_matrix, D_matrix, M, K))
        pess_index, non_pess_index = pessimistic_pareto_adversarial(D_matrix, median_matrix, K)
        print('pess_index:', pess_index, 'non_pess index:', non_pess_index)
        plot_arms_median(median_matrix, D_matrix, M)

        while True:
            median_matrix, D_matrix= satisfy_condition_2(median_matrix, D_matrix, M, K)
            # plot_arms_median(median_matrix, D_matrix, M)
            if check_condition2(median_matrix, D_matrix, M, K):
                break
        condition1= check_condition1(median_matrix, D_matrix, M, K)
        condition2= check_condition2(median_matrix, D_matrix, M, K)
        print('condition 1:',  check_condition1(median_matrix, D_matrix, M, K))
        print('condition 2:' ,check_condition2(median_matrix, D_matrix, M, K))
        pess_index, non_pess_index = pessimistic_pareto_adversarial(D_matrix, median_matrix, K)
        print('pess_index:', pess_index, 'non_pess index:', non_pess_index)
        plot_arms_median(median_matrix, D_matrix, M)
        if condition1 and condition2:
            break
    return  median_matrix, D_matrix

def check_all_conditions(median_matrix, D_matrix, pess_ind, no_pess_ind):
    #check 1
    for j in no_pess_ind:
        satisfied = False
        for i in pess_ind:
            if np.all((median_matrix[j]+ D_matrix[j]) <= (median_matrix[i] - D_matrix[i])):
                satisfied = True
                break
        # print(satisfied)
        if satisfied == False:
            return False

    #check 2
    for i in range(pess_ind.shape[0]):
        for j in range(i+1, pess_ind.shape[0]):
            if np.all((median_matrix[i] - D_matrix[i])  <= (median_matrix[j] + D_matrix[j])):
                return False
            if np.all((median_matrix[j] - D_matrix[j]) <= (median_matrix[i] + D_matrix[i])):
                return False
    return True

class ArmGenerator:
    def __init__(self, K, M, dist_name):
        self.K= K
        self.dist_name= dist_name
        self.M= M
        #
        # D_matrix=  np.zeros([K, M])
        # for i in range(K):
        #     for j in range(M):
        #         bias_class= UnavoidableBias(R, epsilon, advers_name)
        #         D_i_j = bias_class.return_D()
        #         D_matrix[i, j]= D_i_j
        # self.D_matrix= D_matrix

    def create_medians(self, spread):
        # create suboptimal
        D_matrix= self.D_matrix
        K, M = self.K, self.M
        while True:
            median_matrix = np.random.rand(K, M) * spread

            while True:
                median_matrix, D_matrix = add_gap_to_satisfy_condition1(median_matrix, D_matrix, M, K)
                # plot_arms_median(median_matrix, D_matrix, M)
                if check_condition1(median_matrix, D_matrix, M, K):
                    break

            while True:
                median_matrix, D_matrix = satisfy_condition_2(median_matrix, D_matrix, M, K)
                # plot_arms_median(median_matrix, D_matrix, M)
                if check_condition2(median_matrix, D_matrix, M, K):
                    break
            condition1 = check_condition1(median_matrix, D_matrix, M, K)
            condition2 = check_condition2(median_matrix, D_matrix, M, K)

            if condition1 and condition2:
                break

        self.median_matrix= median_matrix

    def create_medians_2obj(self, spread, num_opt):
        D_matrix= self.D_matrix
        K, M = self.K, self.M
        median_matrix= np.zeros([K, M])
        for i in range(num_opt):
            if i == 0:
                median_vec= np.random.rand(M) * spread
            if i!=0:
                median_vec= np.zeros([2,])
                lim1= median_matrix[i-1,0] + D_matrix[i-1, 0]
                lim2= median_matrix[i-1, 1]- D_matrix[i-1, 1]
                median_vec[0] = lim1 + D_matrix[i, 0] + np.random.rand(1)*spread/10
                median_vec[1] = lim2 - D_matrix[i, 1]  - np.random.rand(1)*spread/10

            median_matrix[i, :] = median_vec[:]

        for j in range(K-num_opt):
            median_vec = np.zeros([2, ])
            ind_dom= np.random.choice(num_opt)
            median_vec[0] = median_matrix[ind_dom, 0] - D_matrix[ind_dom, 0] - D_matrix[num_opt + j, 0] - np.random.rand(1)*spread/10
            median_vec[1] = median_matrix[ind_dom, 1] - D_matrix[ind_dom, 1] - D_matrix[num_opt+ j , 1] - np.random.rand(1)*spread/10
            median_matrix[num_opt+j , : ] = median_vec[:]
        self.median_matrix  = median_matrix

    def create_medians_2obj_new_algo(self, reward_min, reward_max):
        K, M = self.K, self.M
        median_matrix= np.random.uniform(reward_min,reward_max,(K, M))
        self.median_matrix  = median_matrix


    def create_samples(self, arm_ind, n):
        median_matrix= self.median_matrix
        M= self.M
        if self.dist_name == 'Gaussian':
            samples= np.random.multivariate_normal(median_matrix[arm_ind],cov= np.eye(M, M), size= n)
        return samples

    def create_samples2(self, arm_ind, obj_ind, N, epsilon):
        indicator = np.random.binomial(1, epsilon, size= N)
        median_matrix= self.median_matrix
        contamination= 1000
        if self.dist_name == 'Gaussian':
            samples= np.random.normal(median_matrix[arm_ind][obj_ind], size= N, scale= 1.0) \
                     * ( 1-indicator) + indicator * contamination
        return samples

    def create_samples3(self, arm_ind, obj_ind, N, epsilon, pareto_inds,std):
        indicator = np.random.binomial(1, epsilon, size= N)
        median_matrix= self.median_matrix
        # contam_amp = 1000
        contam_amp = 1
        if arm_ind in pareto_inds:
            contamination= -contam_amp
        else:
            contamination = contam_amp

        if self.dist_name == 'Gaussian':
            samples= np.random.normal(median_matrix[arm_ind][obj_ind], size= N, scale= std) \
                     * ( 1-indicator) + indicator * contamination
        return samples

    def create_samples_diabetes(self, arm_ind, obj_ind, N, epsilon, sigma):
        indicator = np.random.binomial(1, epsilon, size= N)
        median_matrix= self.y
        # contam_amp = 1000
        # contam_amp = np.random.uniform(20, 50)
        # contamination = -contam_amp

        contamination = np.random.uniform(-50, 50)

        if self.dist_name == 'Gaussian':
            samples= np.random.normal(median_matrix[arm_ind][obj_ind], size= N, scale= sigma) \
                     + indicator * contamination
        return samples


    def load_llvm(self, file_name):
        llvm_dict= pickle_read(file_name)
        self.y = llvm_dict['y']
        self.x= llvm_dict['x']
        self.sample_inds_dict= llvm_dict['sample_inds_dict']

        median_matrix= np.zeros([0,2])
        for arm in self.sample_inds_dict:
            mean_arm = np.mean(self.y[self.sample_inds_dict[arm], :], axis= 0, keepdims= True)
            median_matrix = np.append(median_matrix, mean_arm, axis = 0)

        self.median_matrix = median_matrix

        std_matrix = np.zeros([0,2])
        for arm in self.sample_inds_dict:
            std_arm = np.std(self.y[self.sample_inds_dict[arm], :], axis= 0, keepdims= True)
            std_matrix = np.append(std_matrix, std_arm, axis = 0)
        self.std_matrix= std_matrix

    def load_diabetes(self, file_name):
        diabetes_dict = pickle_read(file_name)
        self.y =diabetes_dict['y']
        self.x= diabetes_dict['x']
        self.median_matrix  = self.y

    def create_samples_llvm(self, arm_ind, obj_ind, N, epsilon):
        contamination = 1000
        indicator = np.random.binomial(1, epsilon, size= N)
        sample_inds= self.sample_inds_dict[arm_ind]
        random_sample_inds= sample_inds[np.random.choice(len(sample_inds), size= N, replace= True)]
        true_samples =self.y[random_sample_inds][:, obj_ind]
        corrupted_samples = true_samples * (1- indicator) + indicator* contamination

        return corrupted_samples

    def create_samples_llvm2(self, arm_ind, obj_ind, N, epsilon, pareto_inds):
        # contam_amp = 1000
        # contam_amp = 1
        contam_amp = 10
        if arm_ind in pareto_inds:
            contamination= -contam_amp
        else:
            contamination = contam_amp
        indicator = np.random.binomial(1, epsilon, size= N)
        sample_inds= self.sample_inds_dict[arm_ind]
        random_sample_inds= sample_inds[np.random.choice(len(sample_inds), size= N, replace= True)]
        true_samples =self.y[random_sample_inds][:, obj_ind]
        corrupted_samples = true_samples * (1- indicator) + indicator* contamination

        return corrupted_samples

def find_pessimistic_and_eliminate(arm_dict, M, D):
    empirical_median_matrix = np.zeros([0, M])
    existing_arm_index = np.zeros([0, ])
    U_vec= np.zeros([0, ])
    for arm in arm_dict:
        empirical_median_vec= np.zeros([1, M])
        U= arm_dict[arm]['Ui']
        U_vec= np.append(U_vec, [U], axis= 0)
        for j in range(M):
            empirical_median_vec[0, j] = arm_dict[arm]['mi_hat'][j]
        empirical_median_matrix= np.append(empirical_median_matrix, empirical_median_vec, axis= 0)
        existing_arm_index= np.append(existing_arm_index, int(arm))
    # D_matrix_existing= D_matrix[existing_arm_index]
    lower_confidence=  empirical_median_matrix- np.expand_dims(U_vec, axis=1) - D
    # plot_arms_median(empirical_median_matrix, np.repeat(np.expand_dims(U_vec, axis=1), M, axis=1), M )
    pess_ind, non_pess_ind= pareto(lower_confidence, K= lower_confidence.shape[0])
    # pess_front= (empirical_median_matrix- np.expand_dims(U_vec, axis=1))[pess_ind]

    dominated= np.zeros([len(existing_arm_index), 1])
    non_dominated= np.ones([len(existing_arm_index), 1])
    for i, arm in enumerate(existing_arm_index):
        median_vec= empirical_median_matrix[i]
        U= U_vec[i]
        for _pess_ind in pess_ind:
            median_vec_pess= empirical_median_matrix[_pess_ind]
            U_pess= U_vec[_pess_ind]
            if _pess_ind == i:
                pass
            elif np.all(median_vec_pess- U_pess -D>= (median_vec + D +U)):
                dominated[i, 0] = 1
                non_dominated[i, 0] = 0

    if len(dominated.nonzero()[0]) == 0:
        return None
    else:
        return existing_arm_index[dominated.nonzero()[0]]

def test_arm_generator():
    K= 10
    M= 3
    dist_name= 'Gaussian'
    advers_name= 'prescient'
    epsilon =0.2
    R= lambda x: x ** 2

    arms= ArmGenerator(K, M, dist_name , epsilon, advers_name, R)
    arms.create_medians()

    #check pessimistic pareto
    print(arms.median_matrix- arms.D_matrix)
    print(pessimistic_pareto_adversarial(arms.D_matrix, arms.median_matrix, K))

    #check condition 1
    print(pessimistic_pareto_adversarial(arms.D_matrix, arms.median_matrix, K))
    print(arms.median_matrix- arms.D_matrix)
    print(arms.median_matrix + arms.D_matrix)