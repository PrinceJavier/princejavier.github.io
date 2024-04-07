import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc, rcParams
from IPython.display import HTML

rc('animation', html='jshtml')
rcParams['animation.html'] = 'jshtml'

shape = (40, 40)
n_iter = 200
entropies = []

# upper half of matrix is 1 the lower half is 0
M = np.zeros(shape)
M[:M.shape[0] // 2] = 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# remove ticks from ax1
ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_ylim(0, 650)
ax2.set_xlim(-10, n_iter)
ax2.set_xlabel("Iterations", fontsize=12)
ax2.set_ylabel("Entropy", fontsize=12)

matrix = ax1.imshow(M, cmap='Blues', aspect="auto")
line, = ax2.plot([], [], lw=2, color='r')
charts = [matrix, line]

# get a dictionary of all values and coordinates in the matrix
def get_coords(matrix):
    dict = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            dict[(i, j)] = matrix[i][j]
    return dict

def init():    

    # upper half of matrix is 1 the lower half is 0
    M = np.zeros(shape)
    M[:M.shape[0] // 2] = 1

    charts[0].set_data(M)
    charts[1].set_data([0, 1], [0, 1])
    return charts


def entropy(M):
    # calculate the count of 1s in the top half vs bottom half of the matrix
    # calculate the number of combinations of 1s in the top half vs bottom half of the matrix

    # the total number of 0s and 1s in 1/2 of the matrix
    # imagine this to be the total number of indices to choose from
    n = M.shape[0] * M.shape[1] // 2 

    # count of 1s in the top half of the matrix
    # imagine this as the indices selected to be 1s
    r = int(M[:M.shape[0] // 2].sum())

    # number of combinations
    num_combs = np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r))

    # get the log to get entropy
    entropy = np.log(num_combs)

    # this is same value when bottom half is considered
    return entropy

def animate(i):

    global entropies

    # get coordinates
    coords = get_coords(M)    

    # list of unswapped coords
    coords_unswapped = list(set(coords.keys()))
    # print(coords_unswapped)

    while len(coords_unswapped) > 0:

        # get coordinates
        coords = get_coords(M)

        # randomly pick a coord index
        coord_idx = np.random.choice(list(range(len(coords_unswapped))), 1)[0]

        # get the coord
        coord_1 = coords_unswapped[coord_idx]

        # get the value
        val_1 = coords[coord_1]

        # get an adjacent coord by for loop either along row or column randomly
        idx = np.random.choice([0, 1]) # 0 - row, 1 - column

        # get the coord x or y
        coord_1b = coord_1[idx]

        # get the adjacent coord x or y
        coord_2b = coord_1b + np.random.choice([-1, 1])

        # if coord_2b is out of bounds, then swap with the opposite side
        if coord_2b < 0:
            coord_2b = M.shape[0] - 1
        elif coord_2b > M.shape[0] - 1:
            coord_2b = 0

        # get coord 2 to swap
        if idx == 0:
            coord_2 = (coord_2b, coord_1[1])
        else:
            coord_2 = (coord_1[0], coord_2b)

        # get the value of coord 2
        val_2 = coords[coord_2]

        # remove coord_1 and coord_2 from coords_unswapped
        if coord_1 in coords_unswapped:
            coords_unswapped.remove(coord_1)
        if coord_2 in coords_unswapped:
            coords_unswapped.remove(coord_2)

        # print(coord_1, coord_2)
        # print(val_1, val_2)

        # swap the values in the Matrix
        M[coord_1] = val_2
        M[coord_2] = val_1


    # entropy plot
    x = np.arange(0, i+1)
    entropies += [entropy(M)]

    charts[0].set_data(M)
    charts[1].set_data(x, entropies)
    return charts

    