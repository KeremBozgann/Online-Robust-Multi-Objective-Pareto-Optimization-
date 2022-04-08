import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.patches as mpatches
from arm_distribution_generator import pessimistic_pareto_adversarial, pareto
import seaborn as sns

def plot_reward_medians(arms, K, save):
    P_ind, _= pessimistic_pareto_adversarial(arms.D_matrix, arms.median_matrix, K)
    fig,ax= plt.subplots(1)
    ax.scatter(arms.median_matrix[:, 0], arms.median_matrix[:, 1], color= 'blue')
    ax.scatter(arms.median_matrix[P_ind, 0], arms.median_matrix[P_ind, 1], color= 'red', label= 'Pareto optimal points')

    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_title('synthetic gaussian example')
    ax.legend()
    for i in range((arms.median_matrix).shape[0]):

        rect = mpatches.Rectangle((arms.median_matrix[i, 0] - arms.D_matrix[i, 0],
                                   arms.median_matrix[i, 1] - arms.D_matrix[i, 1]),
                                  2 * arms.D_matrix[i, 0],
                                  2 * arms.D_matrix[i, 1],
                                  fill=False,
                                  color="purple",
                                  linewidth=1)
        plt.gca().add_patch(rect)
    if save:
        plt.savefig('figure1.pdf')
    plt.show()

def plot_reward_medians_new_algo(arms, K, D, save):
    P_ind, _= pareto(arms.median_matrix, K)
    fig,ax= plt.subplots(1)
    ax.scatter(arms.median_matrix[:, 0], arms.median_matrix[:, 1], color= 'blue')
    ax.scatter(arms.median_matrix[P_ind, 0], arms.median_matrix[P_ind, 1], color= 'red', label= 'Pareto optimal points')

    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_title('synthetic gaussian example')
    ax.legend()
    for i in range((arms.median_matrix).shape[0]):

        rect = mpatches.Rectangle((arms.median_matrix[i, 0] - D,
                                   arms.median_matrix[i, 1] - D),
                                  2 * D,
                                  2 * D,
                                  fill=False,
                                  color="purple",
                                  linewidth=1)
        plt.gca().add_patch(rect)
    if save:
        plt.savefig('figure_median_rewards.pdf')
    plt.show()

def plot_pred_ratios(epsilon_list, correct_pred_mean_auer, correct_pred_std_auer, correct_pred_mean_robust, correct_pred_std_robust):
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

    fig, ax = plt.subplots(1)

    ax.plot(epsilon_list, correct_pred_mean_auer, color= sns.color_palette("bright")[0],linestyle='dashed', marker='s', markerfacecolor='none',
            label='Auer algorithm1',linewidth=1 )

    ax.plot(epsilon_list, correct_pred_mean_robust, color= sns.color_palette("dark")[2],linestyle='dashed', marker='o', markerfacecolor='none',
            label='Ours', markeredgecolor=sns.color_palette("dark")[2], linewidth=1)

    legend = ax.legend(fontsize=9, loc='upper left', frameon=True,
                       facecolor='white', edgecolor="black")
    legend.get_frame().set_alpha(None)

    ax.grid(b=True, which='major', color='black', alpha=0.5, linestyle='--', linewidth=0.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Average ratio of successfull runs')

    ax.set_ylim(top= 1.2)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['xtick.major.pad'] = '5'
    plt.rcParams['ytick.major.pad'] = '5'
    plt.savefig('figure2.pdf', bbox_inches='tight')
    plt.show()

def plot_sample_nums(epsilon_list, total_samp_mean_auer, total_samp_std_auer,
                     total_samp_mean_robust, total_samp_std_robust):


    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

    fig, ax = plt.subplots(1)

    ax.plot(epsilon_list, total_samp_mean_auer, color= sns.color_palette("bright")[0],linestyle='dashed', marker='s', markerfacecolor='none',
            label='Auer algorithm1',linewidth=1)
    ax.errorbar(epsilon_list, total_samp_mean_auer, yerr=2 * total_samp_std_auer,  ls= 'none', capsize=5, alpha=1,
                color= sns.color_palette("bright")[0],ecolor=sns.color_palette("bright")[0], elinewidth= 1, capthick=1)

    ax.plot(epsilon_list, total_samp_mean_robust, color= sns.color_palette("dark")[2],linestyle='dashed', marker='o', markerfacecolor='none',
            label='Ours', markeredgecolor=sns.color_palette("dark")[2], linewidth=1)

    ax.errorbar(epsilon_list, total_samp_mean_robust, yerr= 2 * total_samp_std_robust,  ls= 'none',capsize=5, alpha=1,
                color= sns.color_palette("dark")[2],ecolor= sns.color_palette("dark")[2],  elinewidth= 1,  capthick=1)

    legend = ax.legend(fontsize=9, loc='upper left', frameon=True,
                       facecolor='white', edgecolor="black")
    legend.get_frame().set_alpha(None)

    ax.grid(b=True, which='major', color='black', alpha=0.5, linestyle='--', linewidth=0.5)

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Average number of samples')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['xtick.major.pad'] = '5'
    plt.rcParams['ytick.major.pad'] = '5'
    plt.savefig('figure3.pdf', bbox_inches='tight')
    plt.show()


def plot_medians_in_objective_space(median_matrix, lims= None):

    fig, ax = plt.subplots(1)
    ax.scatter(median_matrix[: , 0], median_matrix[: , 1])
    if lims is not None:
        ax.set_xlim(lims[0][0], lims[0][1])
        ax.set_ylim(lims[0][0], lims[0][1])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['xtick.major.pad'] = '5'
    plt.rcParams['ytick.major.pad'] = '5'
    plt.savefig('objective_space.pdf', bbox_inches='tight')
    plt.close()