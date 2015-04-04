
import pickle
#from matplotlib import pyplot as plt

f = open("friendship_2_random.pkl")
f2r = pickle.load(f)
f.close()
f = open("friendship_2_familiar.pkl")
f2f = pickle.load(f)
f.close()


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
