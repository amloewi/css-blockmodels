
import sys

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import statsmodels.formula.api as sm
import pandas as pd

import blockmodels as bm


a, f, ca, cf, sa, sf = bm.read_css()

ca = np.array(ca, dtype=bool)
cf = np.array(cf, dtype=bool)


def majority_accuracy(data, edge, positive=True, edge_given_obs=True, trials=1):
    N = len(edge)
    l0 = []
    ix = range(N)
    for t in range(trials):
        if trials > 1: np.random.shuffle(ix)
        l = []
        for i in range(N):





            # Include the first i elements from the shuffled 'ix'
            # FUNNY EDGE CASE: when everybody disagrees perfectly,
            # EVERYTHING exists, because 50/50 breaks 'yes'
            obs = np.array(np.sum(data[ix[:i+1]], axis=0) >= (i+1)/2., dtype=bool)
            # if i in [8,9,10]: print np.int64(obs)
            # Do I want p(edge|obs) OR p(obs|edge)?






            denom = obs if edge_given_obs else edge
            if positive:
                # Ex: p(e=1|o=1) = p(e=1,o=1)/p(o=1)
                acc = np.mean( obs &  edge)/np.mean( denom)
            else:
                acc = np.mean(~obs & ~edge)/np.mean(~denom)
            l.append(acc)
        l0.append(l)
    return l0

def neighbor_accuracy(data, edge, positive=True, edge_given_obs=True, trials=1):
    """
    """
    N = len(edge)
    l0 = []
    ix = range(N)
    for t in range(trials):
        # ix contains the INDICES of the people
        if trials > 1: np.random.shuffle(ix)
        l = []
        # i is the NUMBER of people we ask
        for i in range(N):

            # Include the first i elements from the shuffled 'ix'
            people = ix[:i+1]
            obs = np.zeros((N,N), dtype=bool)

            # Iterating over all the relationships.
            for p1 in range(N):
                for p2 in range(N):

                    # This is SELF-PROCLAIMED neighbors
                    neighbors = []
                    # Iterating over all the people -- who CLAIMS to be a neighbor?
                    for n in range(N):
                        # Did more than 0 people say they had a relationship?
                        yeah = sum([data[n,p1,n],
                                    data[n,n,p1],
                                    data[n,p2,n],
                                    data[n,n,p2]])
                        if n in people and yeah:
                            neighbors.append(n)
                    print 'edge: ', (p1, p2)
                    print 'neighbors: ', neighbors
                    # What do the neighbors say?
                    votes = sum([data[n,p1,p2] for n in neighbors])
                    obs[p1,p2] = neighbors and (votes >= len(neighbors)/2.)

            # Do I want p(edge|obs) OR p(obs|edge)?
            denom = obs if edge_given_obs else edge
            if positive:
                acc = np.mean( obs &  edge)/np.mean( denom)
            else:
                acc = np.mean(~obs & ~edge)/np.mean(~denom)
            l.append(acc)
        l0.append(l)
    return l0

def block_accuracy(data, edge, groups, positive=True, edge_given_obs=True, trials=1):
        l0 = []
        N = len(edge)
        ix = range(N)
        for t in range(trials):
            if trials > 1: np.random.shuffle(ix)
            l = []
            for i in range(N):



                # Include the first i elements from the shuffled 'ix'
                # FUNNY EDGE CASE: when everybody disagrees perfectly,
                # EVERYTHING exists, because 50/50 breaks 'yes'
                in_block = [p for p in ix[:i+1] if groups[p] == groups[i]]
                obs = np.array(np.sum(data[in_block], axis=0) >= len(in_block)/2., dtype=bool)
                # if i in [8,9,10]: print np.int64(obs)
                # Do I want p(edge|obs) OR p(obs|edge)?



                denom = obs if edge_given_obs else edge
                if positive:
                    # Ex: p(e=1|o=1) = p(e=1,o=1)/p(o=1)
                    acc = np.mean( obs &  edge)/np.mean( denom)
                else:
                    acc = np.mean(~obs & ~edge)/np.mean(~denom)
                l.append(acc)
            l0.append(l)
        return l0


def accuracy(data, edge, positive=True, edge_given_obs=True, trials=1,
                                threshold=0.5, groups=None, kind="majority"):
    all_accuracies = []
    N = len(edge)
    people = range(N)
    for t in range(trials):
        if trials > 1: np.random.shuffle(people)
        loop_accuracies = []
        for num_people in range(N):
################################################################################
            sample = people[:num_people+1]
            if kind=="majority":
                # Include the first i elements from the shuffled 'people'
                # FUNNY EDGE CASE: when everybody disagrees perfectly,
                # EVERYTHING exists, because 50/50 breaks 'yes'
                obs = np.sum(data[sample], axis=0)

                for p1 in range(N): #np.ndindex((N,N)):
                    for p2 in range(N):

                        sender   = (p1, p1, p2) #(e[0], e[0], e[1])
                        receiver = (p2, p1, p2) #(e[1], e[0], e[1])
                        # Remove their opinions
                        obs[(p1,p2)] -= data[sender]
                        obs[(p2,p2)] -= data[receiver]
                        # And divide by the number of observers
                        # who WEREN'T THEM.
                        penalty = (p1 in sample) + (p2 in sample)
                        # now its: #opinions-not-theirs /#people-not-them
                        obs[(p1,p2)] /= num_people - penalty

                # Before, this was (i+1)/2., because #>n ... but now it's -=2.
                obs = np.array(obs >= threshold, dtype=bool)
            elif kind=="neighbor":
                obs = np.zeros((N,N))#, dtype=bool)
                # Iterating over all the relationships.
                for p1 in range(N):
                    for p2 in range(N):
                        # This is SELF-PROCLAIMED neighbors
                        neighbors = []
                        # Iterating over all the people -- who CLAIMS to be a neighbor?
                        for n in range(N):
                            # Did more than 0 people say they had a relationship?
                            yeah = sum([data[n,p1,n],
                                        data[n,n,p1],
                                        data[n,p2,n],
                                        data[n,n,p2]])
                            if yeah and n in sample and n not in [p1,p2]:
                                neighbors.append(n)
                        # What do the neighbors say?
                        # Edge PARTICIPANTS have already been removed ^.
                        votes = sum([data[n,p1,p2] for n in neighbors])
                        score = float(votes)/len(neighbors) if neighbors else 0
                        obs[p1,p2] = score >= threshold
                # And finally ...
                obs = np.array(obs, dtype=bool)

            elif kind=="block":
                obs = np.zeros((N,N))#, dtype=bool)
                for p1, p2 in np.ndindex((N,N)):
                    # You are 'in block' if you share ONE endpoint of the edge,
                    # because otherwise you have NO observers.
                    in_block=[p for p in sample if
                                # You're in one of the edge's groups ..
                                groups[p] in (groups[p1], groups[p2]) and
                                # But you're NOT in the edge itself!
                                p != p1 and p != p2]
                    votes = sum([data[n,p1,p2] for n in in_block])
                    score = float(votes)/len(in_block) if in_block else 0
                    obs[p1,p2] = score >= threshold
                # Cast as boolean for the bit operations
                obs = np.array(obs, dtype=bool)
################################################################################
            denom = obs if edge_given_obs else edge
            if positive:
                # Ex: p(e=1|o=1) = p(e=1,o=1)/p(o=1)
                acc = np.mean( obs &  edge)/np.mean( denom)
            else:
                acc = np.mean(~obs & ~edge)/np.mean(~denom)
            loop_accuracies.append(acc)
        all_accuracies.append(loop_accuracies)
    return all_accuracies


def test_majority_accuracy(m):
    # right ...
    x = np.array(bm.blocked_matrix(21,2), dtype=bool)
    data = np.array([x if i < 5 else ~x for i in range(21)])
    # data, truth, ptp/ptn, e|o, trials
    return majority_accuracy(data, np.ones((21,21), dtype=bool), True, False)

def test_neighbor_accuracy():
    # A simple tree
    edges = [(0,1),(1,2),(2,3),(3,4),(2,5),(5,6)]
    g = nx.Graph()
    g.add_edges_from(edges)
    g = np.array(nx.to_numpy_matrix(g), dtype=bool)
    h = np.array([g for i in range(len(g))])
    print 'g: \n', np.int64(g)
    return accuracy(h, g, kind="neighbor")

def test_block_accuracy(m):
    pass


def plotem(d):
    for k, v in d.items():
        for i in v:
            plt.plot(i)
        plt.ylim([0,1])
        plt.savefig(k+".png")
        plt.close()

if __name__ == '__main__':
    # THEY'RE DIFFERENT BECAUSE I DIDN'T HAVE '>=' BEFORE, JUST '>'

    #data, edge, positive=True, edge_given_obs=True, trials=1,
    #                            threshold=0.5, groups=None, kind="majority"):
        # ConsensusFriendship(/Advice)BlockmodelCorrected
    ga = bm.cabc.groups
    gf = bm.cfbc.groups

    margs = [("majority_p(edge=1|obs=1)_advice_shuffles",     a, ca, True,  True,  100, 0.5, None, "majority"),
             ("majority_p(edge=0|obs=0)_advice_shuffles",     a, ca, False, True,  100, 0.5, None, "majority"),
             ("majority_p(edge=1|obs=1)_friendship_shuffles", f, cf, True,  True,  100, 0.5, None, "majority"),
             ("majority_p(edge=0|obs=0)_friendship_shuffles", f, cf, False, True,  100, 0.5, None, "majority"),

             ("majority_p(obs=1|edge=1)_advice_shuffles",     a, ca, True,  False, 100, 0.5, None, "majority"),
             ("majority_p(obs=0|edge=0)_advice_shuffles",     a, ca, False, False, 100, 0.5, None, "majority"),
             ("majority_p(obs=1|edge=1)_friendship_shuffles", f, cf, True,  False, 100, 0.5, None, "majority"),
             ("majority_p(obs=0|edge=0)_friendship_shuffles", f, cf, False, False, 100, 0.5, None, "majority")]

    nargs = [("neighbor_p(edge=1|obs=1)_advice_shuffles",     a, ca, True,  True,  100, 0.5, None, "neighbor"),
             ("neighbor_p(edge=0|obs=0)_advice_shuffles",     a, ca, False, True,  100, 0.5, None, "neighbor"),
             ("neighbor_p(edge=1|obs=1)_friendship_shuffles", f, cf, True,  True,  100, 0.5, None, "neighbor"),
             ("neighbor_p(edge=0|obs=0)_friendship_shuffles", f, cf, False, True,  100, 0.5, None, "neighbor"),

             ("neighbor_p(obs=1|edge=1)_advice_shuffles",     a, ca, True,  False, 100, 0.5, None, "neighbor"),
             ("neighbor_p(obs=0|edge=0)_advice_shuffles",     a, ca, False, False, 100, 0.5, None, "neighbor"),
             ("neighbor_p(obs=1|edge=1)_friendship_shuffles", f, cf, True,  False, 100, 0.5, None, "neighbor"),
             ("neighbor_p(obs=0|edge=0)_friendship_shuffles", f, cf, False, False, 100, 0.5, None, "neighbor")]

    bargs = [("block_p(edge=1|obs=1)_advice_shuffles",        a, ca, True,  True,  100, 0.5, ga,   "block"),
             ("block_p(edge=0|obs=0)_advice_shuffles",        a, ca, False, True,  100, 0.5, ga,   "block"),
             ("block_p(edge=1|obs=1)_friendship_shuffles",    f, cf, True,  True,  100, 0.5, gf,   "block"),
             ("block_p(edge=0|obs=0)_friendship_shuffles",    f, cf, False, True,  100, 0.5, gf,   "block"),

             ("block_p(obs=1|edge=1)_advice_shuffles",        a, ca, True,  False, 100, 0.5, ga,   "block"),
             ("block_p(obs=0|edge=0)_advice_shuffles",        a, ca, False, False, 100, 0.5, ga,   "block"),
             ("block_p(obs=1|edge=1)_friendship_shuffles",    f, cf, True,  False, 100, 0.5, gf,   "block"),
             ("block_p(obs=0|edge=0)_friendship_shuffles",    f, cf, False, False, 100, 0.5, gf,   "block")]

    targs = [("majority_p(edge=1|obs=1)_friendship_threshold=3",    f, cf, True,  True,  100, 0.3, None, "majority"),
             ("neighbor_p(edge=1|obs=1)_friendship_threshold=3",    f, cf, True,  True,  100, 0.3, None, "neighbor"),
             ("block_p(edge=1|obs=1)_friendship_threshold=3",       f, cf, True,  True,  100, 0.3, gf,   "block"),
             ("majority_p(edge=1|obs=1)_friendship_threshold=4",    f, cf, True,  True,  100, 0.4, None, "majority"),
             ("neighbor_p(edge=1|obs=1)_friendship_threshold=4",    f, cf, True,  True,  100, 0.4, None, "neighbor"),
             ("block_p(edge=1|obs=1)_friendship_threshold=4",       f, cf, True,  True,  100, 0.4, gf,   "block"),
             ("majority_p(edge=1|obs=1)_friendship_threshold=6",    f, cf, True,  True,  100, 0.6, None, "majority"),
             ("neighbor_p(edge=1|obs=1)_friendship_threshold=6",    f, cf, True,  True,  100, 0.6, None, "neighbor"),
             ("block_p(edge=1|obs=1)_friendship_threshold=6",       f, cf, True,  True,  100, 0.6, gf,   "block"),
             ("majority_p(edge=1|obs=1)_friendship_threshold=7",    f, cf, True,  True,  100, 0.7, None, "majority"),
             ("neighbor_p(edge=1|obs=1)_friendship_threshold=7",    f, cf, True,  True,  100, 0.7, None, "neighbor"),
             ("block_p(edge=1|obs=1)_friendship_threshold=7",       f, cf, True,  True,  100, 0.7, gf,   "block")]




    # dm = {a[0]: accuracy(*a[1:]) for a in margs}
    # dn = {a[0]: accuracy(*a[1:]) for a in nargs}
    db = {a[0]: accuracy(*a[1:]) for a in bargs}
    dt = {a[0]: accuracy(*a[1:]) for a in targs}

    # plotem(dm)
    # plotem(dn)
    plotem(db)
    plotem(dt)
    #between = [(10,17),(16,11),(4,20),(4,1),(5,1),(16,20),(20,16)]

    sys.exit()

    N = len(cf)
    data = []
    for e in np.ndindex(cf.shape):
        data.append([cf[e]] + list(f[:,e[0],e[1]]))
    df = np.array(data) #pd.DataFrame(data)
    lr = sm.Logit(df[:,0], df[:,range(1,21+1)], Iterations=50).fit()
    print lr.summary()

    obs = np.zeros((N,N))
    predicted = logit(lr.fittedvalues)
    for i, e in enumerate(np.ndindex(cf.shape)):
        obs[e] = predicted[i]
    for i in np.linspace(.45,.55,10):
        positive = np.sum((obs>=i) & (cf==1))/float(np.sum(cf==1))
        negative = np.sum((obs< i) & (cf==0))/float(np.sum(cf==0))
        #print 'Before adding consensus: {:.3f}'.format(before)
        print "Threshold: ", i
        print '% True Positive: ', positive
        print '% True Negative: ', negative

    # .505 works as ~the best boundary for a simple LR model.
    for i in np.linspace(0.08,.11,10):
        obs = np.array(np.sum(f, axis=0)/N > i, dtype=float)
        positive = np.sum((obs>=i) & (cf==1))/float(np.sum(cf==1))
        negative = np.sum((obs< i) & (cf==0))/float(np.sum(cf==0))
        print "Threshold: ", i
        print '% True Positive: ', positive
        print '% True Negative: ', negative


def roc(f, data):

    tpr = []
    fpr = []
    thresholds = np.linspace(0,1)
    for t in thresholds:
        tp, fp = f(data)
    #          X, Y
    plt.plot(fpr,tpr)
    bm.show()

#if __name__ == '__main__':
#    pass


cfb = bm.blockmodel(cf, 2)
cfbc = bm.blockmodel(cf, 2, corrected=True)
tp, fp, tn, fn = em.count_errors(f, cf, cfb.groups, 2)
pfp, pfn = em.estimate_error_rates(tp, fp, tn, fn)

# And this, now, is ... p(o=1|e=0)
##################################
# In [309]: pfp
# Out[309]:
# array([[[ 0.025     ,  0.04357298],
#         [ 0.03222222,  0.03806584]],
#
#        [[ 0.01354167,  0.06699346],
#         [ 0.0675    ,  0.08179012]]])
#
# In [310]: pfn
# Out[310]:
# array([[[ 0.77777778,  0.64814815],
#         [ 0.77777778,  0.78703704]],
#
#        [[ 1.        ,  0.45833333],
#         [ 0.70833333,  0.51157407]]])
#
# This CAN'T be right --
#



# This was p(e|o)
#
# In [1707]: pfp
# Out[1707]:
# array([[[ 0.39800995,  0.75824176],
#         [ 0.95555556,  0.35294118]],
#
#        [[ 0.35616438,  0.65625   ],
#         [ 0.925     ,  0.31677019]]])
#
# In [1708]: pfn
# Out[1708]:
# array([[[ 0.17396746,  0.03766105],
#         [ 0.00594059,  0.10288809]],
#
#        [[ 0.23271665,  0.0466893 ],
#         [ 0.00683761,  0.07521368]]])
# VERSUS
# 0.51 pfp overall, and 0.08 pfn overall



# TODO
# compare the graphs of majority vs. consensus
# compare the groups of both as well
# compare the groups from 2d (of ... what?) vs. 3d modeling
# have a better way to assess an edge than majority.
# And oh yeah, TEST the accuracy functions above.







# majap = majority_accuracy(a, ca)
# majan = majority_accuracy(a, ca, positive=False)
# majfp = majority_accuracy(f, cf)
# majfn = majority_accuracy(f, cf, positive=False)
#
# for a, b in [("majap", majap), ("majan", majan), ("majfp", majfp), ("majfn", majfn)]:
#

# # Advice, positive: p(v=1, e=1)/p(e=1)
# for i in range(21):
#     maj = np.array(np.sum(a[range(i)], axis=0) > i/2., dtype=bool)
#     ptp = np.mean(maj & ca)/np.mean(ca)
#     majap.append(ptp)
#
# # Advice, negative p(v=0, e=0)/p(e=0)
# for i in range(21):
#     maj = np.array(np.sum(a[range(i)], axis=0) > i/2., dtype=bool)
#     ptn = np.mean(~maj & ~ca)/np.mean(~ca)
#     majan.append(ptn)
#
# # Friendship, positive p(v=1, e=1)/p(e=1)
# for i in range(21):
#     maj = np.array(np.sum(f[range(i)], axis=0) > i/2., dtype=bool)
#     ptp = np.mean(maj & cf)/np.mean(cf)
#     majfp.append(ptp)
#
# # Friendship, negative p(v=0, e=0)/p(e=0)
# for i in range(21):
#     maj = np.array(np.sum(f[range(i)], axis=0) > i/2., dtype=bool)
#     ptn = np.mean(~maj & ~cf)/np.mean(~cf)
#     majfn.append(ptn)
#
# # I WANT TO DO THE SAME FOR ... NEIGHBORS/NON-NEIGHBORS
#
#
# plt.plot(majap)
# plt.ylim([0,1])
# plt.savefig("majap.png")
# plt.close()
#
# plt.plot(majan)
# plt.ylim([0,1])
# plt.savefig("majan.png")
# plt.close()
#
# plt.plot(majfp)
# plt.ylim([0,1])
# plt.savefig("majfp.png")
# plt.close()
#
# plt.plot(majfn)
# plt.ylim([0,1])
# plt.savefig("majfn.png")
# plt.close()


# # Friendship, positive
# for i in range(21):
#     maj = np.int64(np.sum(f[range(i)], axis=0) > i/2.)
#     majfp.append(np.mean(np.int64(maj==cf)==1))
