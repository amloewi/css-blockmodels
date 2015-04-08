
import json

import numpy as np
from matplotlib import pyplot as plt

import blockmodels as bm

# f = open("friendship_2_random.pkl")
# f2r = pickle.load(f)
# f.close()
# f = open("friendship_2_familiar.pkl")
# f2f = pickle.load(f)
# f.close()
f = open("advice_2_random.json")
a2r = json.load(f)
f.close()
# f = open("advice_2_familiar.pkl")
# a2f = json.load(f)
# f.close()

# one_sample['sample'] = sample
# one_sample['edge_accuracy'] = inferred_edge_accuracy
# one_sample['full_groups_accuracy'] = full_data_groups_distance
# one_sample['g_hat_groups_accuracy'] = inferred_network_groups_distance
# one_sample['em_iterations'] = nits
#
# one_sample['b_hat_p'] = b_hat.p.tolist()
# one_sample['full_data_p'] = m.p.tolist()
#
# one_sample['b_groups_pfp'] = pfp_b.tolist()
# one_sample['b_groups_pfn'] = pfn_b.tolist()
#
# one_sample['full_groups_pfp'] = pfp_m.tolist()
# one_sample['full_groups_pfn'] = pfn_m.tolist()


# This looks TOTALLY fucked --
def plot_group_accuracy(x, name, full=False, show=True):

    data = []
    for iteration in x:
        if full:
            data.append([sample["full_groups_accuracy"] for sample in iteration])
        else:
            data.append([sample["g_hat_groups_accuracy"] for sample in iteration])

    plt.plot(np.array(data).T)
    plt.ylim([-.1, 1.1])
    plt.xlabel("num samples")
    plt.ylabel("group similarity")
    plt.savefig(name)

    if show:
        bm.show()


def plot_edge_accuracy(x, name, show=True):

    data = [[],[]]
    for iteration in x:
        data[0].append([sample['edge_accuracy'][0] for sample in iteration])
        data[1].append([sample['edge_accuracy'][1] for sample in iteration])

    #plt.plot(data)
    for i in [0,1]:
        for row in data[i]:
            if i:
                plt.plot(row, color="red")
            else:
                plt.plot(row, color="blue")

    plt.ylim([-.1, 1.1])
    plt.xlabel("num samples")
    plt.ylabel("edge accuracy")
    plt.savefig(name)

    if show:
        bm.show()

def plot_error_rates(x):
    pass
    


if __name__=="__main__":
    plot_group_accuracy(a2r, "a2r_group_accuracy.png")
    # This LOOKS reasonable --
    #plot_edge_accuracy(a2r, "a2r_edge_accuracy.png")



# Try 100 blockmodels -- how long before it gets the highest one?
# Watch the changes in likelihood over EM -- how long it before it hits
# BASICALLY the top?


# So I've still GOT the data, but it'll need to be cast to lists,
# for ... compatibility

# # for each collection scheme,
#     # for each iteration,
#         # for each stage OF the iteration, meaning concrete sample, we have:
#
# sample
# inferred_edge_accuracy
# full_data_groups_distance
# inferred_network_groups_distance
# b_hat
# nits
# m
# pfp_b
# pfn_b
# pfp_n
# pfn_n
#
#
# # WHAT PLOTS DO I WANT?
# # Oh, fuck ...
# parameter estimates versus true -- but for WHICH CASES?
#
# group/edge accuracy over n -- for random and active
