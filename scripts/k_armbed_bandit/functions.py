import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# for visualization of animation
from PIL import Image
from matplotlib import animation
from IPython.display import HTML

# generate q*(a)
def init_q(k=10, random_q_init=True, std=1):

    # initialize means of q
    if random_q_init:
        q_mu = np.random.normal(0, std, k)
    elif not random_q_init:
        q_mu = np.zeros(k)
    return q_mu

# add random walk
def random_walk_q(q_mu, std=0.01):
    delta = np.random.normal(0, std, size=len(q_mu))
    q_mu = q_mu + delta
    return q_mu

# randomly generate N samples for visualization purposes
def viz_r_dist(k, q_mu, N=1000, ax=None):
    r_samples = np.random.normal(q_mu, 1, size=(N, k))
    r_samples = pd.DataFrame(r_samples)
    
    # visualize
    # sns.violinplot(r_samples, inner='quart', color='orange', ax=ax)
    sns.boxplot(r_samples, color='orange', ax=ax, showfliers=False)

def update_Q(Q_old, R, step_size):
    Q_new = Q_old + step_size * (R - Q_old)
    return Q_new

def get_R(q_mu, a, std=1):
    return np.random.normal(q_mu[a], std)

def moving_average(x, w):
    if w == 0:
        return x
    elif w > 0:
        return np.convolve(x, np.ones(w), 'valid') / w

# sample average
# for sample average method, step_size = 1/N(A)

class Bandit:
    def __init__(self, 
                 k_arms, 
                 eps,
                 random_q_init, 
                 stationary_q = True,
                 q_delta_std = 0.01,
                 steps = 10000,
                 step_size = "1/N",
                 random_seed = 42
                 ):
        self.k_arms = k_arms
        self.eps = eps
        self.random_q_init = random_q_init
        self.stationary_q = stationary_q
        self.q_delta_std = q_delta_std
        self.steps = steps
        self.step_size = step_size
        self.random_seed = random_seed

    def run_bandit(self):    
        np.random.seed(self.random_seed)
        
        # tracker of Q(a) and N(a)
        Q_tracker = np.zeros((self.steps, self.k_arms)) # steps x k arms
        N_tracker = np.zeros((self.steps, self.k_arms)) # steps x k arms 
        R_tracker = []
        A_tracker = []
        A_optimal_tracker = []

        # initialize q()
        self.q_mu = init_q(k=self.k_arms, random_q_init=self.random_q_init, std=1)

        for step in range(0, self.steps):
            if not self.stationary_q:
                self.q_mu = random_walk_q(self.q_mu, std=self.q_delta_std)

            exploit = np.random.choice([True, False], p = [1-self.eps, self.eps])
            if exploit:
                # greedy action based on Q
                Q_current = Q_tracker[step, :]
                a = np.random.choice(np.argwhere(Q_current == Q_current.max()).flatten())
                # a = np.argmax(Q_current)
            elif not exploit:
                # explore
                a = np.random.choice(range(self.k_arms))

            # get reward
            R = get_R(self.q_mu, a, std=1)

            # update Q_tracker and N_tracker
            N_tracker[step, a] += 1
            # copy to next step to persist
            N_tracker[step+1, :] = N_tracker[step, :]

            if self.step_size == "1/N":
                self.step_size = 1 / N_tracker[step, a]

            Q_tracker[step, a] = update_Q(Q_tracker[step, a], R, self.step_size)
            # copy to next step to persist
            Q_tracker[step+1, :] = Q_tracker[step, :]

            R_tracker += [R]
            A_tracker += [a]
            A_optimal_tracker += [np.argmax(self.q_mu)] # the correct one

            if step + 1 == self.steps - 1:
                break

        self.Q_tracker = np.array(Q_tracker)
        self.N_tracker = np.array(N_tracker)
        self.R_tracker = np.array(R_tracker)
        self.A_tracker = np.array(A_tracker)
        self.A_optimal_tracker = np.array(A_optimal_tracker)

def plot_learning_curves(stationary_q, random_q_init, q_delta_std,
                         k_arms, n_trials, steps,                         
                         step_sizes = [0.1, "1/N"], eps_list = [0, 0.1, 0.01],
                         w = 0,
                         ):
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)    
    for step_size in step_sizes:    
        for eps in eps_list:

            R_vals = []
            Q_vals = []
            N_vals = []
            A_prop_correct= []

            for trial in range(n_trials):
                bandit = Bandit(k_arms, eps=eps,
                                random_q_init=random_q_init,
                                stationary_q=stationary_q, 
                                q_delta_std=q_delta_std,
                                steps=steps, step_size=step_size,
                                random_seed=trial,
                                )
                bandit.run_bandit()
                
                R_vals += [bandit.R_tracker]
                cnt_correct = (bandit.A_tracker == bandit.A_optimal_tracker)
                A_prop_correct += [cnt_correct]
                Q_vals += [bandit.Q_tracker]
                N_vals += [bandit.N_tracker]


            axs[0].plot(moving_average(np.array(R_vals).mean(axis=0), w), label=f'eps = {eps}; step_size = {step_size}')
            axs[1].plot(moving_average(np.array(A_prop_correct).mean(axis=0), w), label=f'eps = {eps}; step_size = {step_size}')

    axs[0].set_title("Average Return")
    axs[1].set_title("Percent Optimal Actions")

    axs[0].set_xlabel("timestep")
    axs[0].set_ylabel(f"average return (MA-{w})")

    axs[1].set_xlabel("timestep")
    axs[1].set_ylabel(f"pct optimal actions (MA-{w})")

    # axs[0].set_ylim(0, 4)
    # axs[1].set_ylim(0, 1)

    axs[0].legend()

    plt.suptitle(f"Stationary q: {stationary_q}")
    plt.tight_layout()
    sns.despine()