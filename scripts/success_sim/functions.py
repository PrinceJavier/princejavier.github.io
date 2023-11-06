import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc, rcParams
import seaborn as sns
from IPython.display import HTML

rc('animation', html='jshtml')
rcParams['animation.html'] = 'jshtml'

# generate (T,O) for N individuals
N = 100000 # people
M = 100 # ideas per person
T = np.random.normal(size=N, loc=0, scale=1)
O = np.random.power(a=0.16, size=N)
r = np.random.normal(size=N, loc=0, scale=1)
Q = T * O

sum_r = []
for i in range(N):
    rs = np.random.normal(size=M, loc=0, scale=1)
    sum_r += [sum(rs)]

assert len(sum_r) == N

# successes
S = Q * sum_r

# pareto
assert np.around(O[np.argwhere(O > np.percentile(O, q=80))].sum() / O.sum(), 2) == 0.8

# animation
fig, axs = plt.subplots(2, 2, figsize=(15, 8))
axs = axs.flatten()

def animate(q):
    q = q * 5
    [ax.clear() for ax in axs]

    axs[0].set_title(f"Dist of talents for top {q}% most successful")
    axs[1].set_title(f"Dist of successes for top {q}% most talented")
    axs[2].set_title(f"Dist of opportunities for top {q}% most successful")
    axs[3].set_title(f"Dist of opportunities for top {q}% most talented")

    thresh = np.percentile(S, q)
    T_thresh = T[np.argwhere(S > thresh)]
    sns.kdeplot(T_thresh, ax=axs[0], label=q)

    thresh = np.percentile(T, q)
    S_thresh = S[np.argwhere(T > thresh)]
    sns.kdeplot(S_thresh, ax=axs[1], label=q)

    thresh = np.percentile(S, q)
    O_thresh = O[np.argwhere(S > thresh)]
    sns.kdeplot(O_thresh, ax=axs[2], label=q)    

    thresh = np.percentile(T, q)
    O_thresh = O[np.argwhere(T > thresh)]
    sns.kdeplot(O_thresh, ax=axs[3], label=q)        

    return None