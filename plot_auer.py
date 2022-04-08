import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_mean(M, K, arm_dict, P, eliminated, arm_gen):
    combined= {**{**arm_dict, **P}, **eliminated}
    index_list= np.zeros([0, ])
    if M==2:
        empirical_mean_matrix = np.zeros([K, M])
        Beta_vec= np.zeros([K, 1])
        for arm in combined:
            index_list= np.append(index_list, np.array([int(arm)]), axis= 0)
            empirical_mean_vec= np.zeros([1, M])
            beta= combined[arm]['Beta']
            for j in range(M):
                empirical_mean_vec[0,j]= combined[arm]['mi_hat'][j]
            empirical_mean_matrix[int(arm)] = empirical_mean_vec[0, :]
            Beta_vec[int(arm)]= beta

        median_matrix= arm_gen.median_matrix
        D_matrix= arm_gen.D_matrix
        Beta_matrix= np.repeat(Beta_vec, M, axis= 1)
        fig, ax = plt.subplots(1)
        ax.scatter(empirical_mean_matrix[:, 0], empirical_mean_matrix[:, 1], color='red')
        ax.scatter(median_matrix[:, 0], median_matrix[:, 1], color = 'blue')
        for i in range((arm_gen.median_matrix).shape[0]):
            rect = mpatches.Rectangle((arm_gen.median_matrix[i, 0] - arm_gen.D_matrix[i, 0] - Beta_matrix[
                np.where(index_list == i)[0][0], 0],
                                       arm_gen.median_matrix[i, 1] - arm_gen.D_matrix[i, 1] - Beta_matrix[
                                           np.where(index_list == i)[0][0], 1]),
                                      2 * ((arm_gen.D_matrix)[i, 0] + Beta_matrix[np.where(index_list == i)[0][0], 0]),
                                      2 * ((arm_gen.D_matrix)[i, 1] + Beta_matrix[np.where(index_list == i)[0][0], 1]),
                                      fill=False,
                                      color="purple",
                                      linewidth=1)
            plt.gca().add_patch(rect)
            rect = mpatches.Rectangle((arm_gen.median_matrix[i, 0] - arm_gen.D_matrix[i, 0],
                                       arm_gen.median_matrix[i, 1] - arm_gen.D_matrix[i, 1]),
                                      2 * arm_gen.D_matrix[i, 0],
                                      2 * arm_gen.D_matrix[i, 1],
                                      fill=False,
                                      color="orange",
                                      linewidth=1)
            ax.annotate(str(int(index_list[i])), (empirical_mean_matrix[i, 0], empirical_mean_matrix[i, 1]))
            ax.annotate(str(i) + ',t', (arm_gen.median_matrix[i, 0], arm_gen.median_matrix[i, 1]))
            plt.gca().add_patch(rect)
        plt.show()

