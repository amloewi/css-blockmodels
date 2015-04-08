
import sys
import copy
import pdb

import numpy as np
import networkx as nx

import npkmeans as km
import blockmodels as bm
import grouping_distance as gd


def calculate_likelihood(data, groups, G, n, k):

    Po1e0   = np.zeros((k,)*3) # p(o=1,e=0)
    Po1e1   = np.zeros((k,)*3) # p(o=1,e=1)

    counts = {0:np.zeros((k,)*3), # observations for Po1e0
              1:np.zeros((k,)*3)} # observations for Po1e1

    Pe0   = np.zeros((k,)*3) # p(e=0)
    Pe1   = np.zeros((k,)*3) # p(e=1)

    # Iterate over every observation
    for o_ijk in bm.ndindex(data.shape):

        # The elements are Perceiver, Sender, Receiver, to the last two
        # elements form the 2D edge
        e = o_ijk[1:]
        # This is the Block to the which the perceptions belongs --
        # Just the groups of the three elements IN the perception.
        B = tuple(groups[list(o_ijk)])

        # sum_edge sum_people obs*p(obs) (by block)
        Po1e0[B] += data[o_ijk]*(1-G[e]) # p(o=1,e=0) += obs_ijk*p(e_jk=0)
        Po1e1[B] += data[o_ijk]*G[e]     # p(o=1,e=1) += obs_ijk*p(e_jk=1)

        # Is this right?
        counts[data[o_ijk]][B] += 1

        Pe0[B] += (1-G[e]) # p(e_jk=0) += p_i(e_jk=0)
        Pe1[B] +=    G[e]  # p(e_jk=1) += p_i(e_jk=0)

    likelihood = {0:Po1e0/Pe0,   # p(o=1|e=0) = p(o=1,e=0)/p(e=0)
                  1:Po1e1/Pe1}   # p(o=1|e=1) = p(o=1,e=1)/p(e=1)
    # If n_A = 0, we'll have /0 errors in likelihood --
    # replace them as 0?
    for i in [0,1]:
        likelihood[i][np.isnan(likelihood[i])] = np.mean(likelihood[i][~np.isnan(likelihood[i])])
        # Smoothing -- an attempt.
        likelihood[i][likelihood[i] < .1] = 0.1
        likelihood[i][likelihood[i] > .9] = 0.9
        #likelihood[i][counts[i] == 0] = np.mean(likelihood[i][counts[i]!=0])
        #.1 # Pseudocounts, for no data
        # OR, should it be -- um, the average for the accuracy of the OTHER
        # observed groups? Probably a better guess.

        # # HACKED -- is it an identifiability problem?
        # if i:
        #     likelihood[i][likelihood[i] <  .5] = .51
        # else:
        #     likelihood[i][likelihood[i] >= .5] = .49


    # DANGER: ALSO -- what do I do about ... lack of observations?
    # pseudocounts?

    return likelihood


def calculate_posterior(data, groups, Po1_e, p, N, k):

    POe0 = np.ones((N,N)) # p(O,e=0)
    POe1 = np.ones((N,N)) # p(O,e=1)

    for o_ijk in bm.ndindex(data.shape):

        e = o_ijk[1:]
        B = tuple(groups[list(o_ijk)])
        # p(O|e=0)  p(o_ijk=1 | e=0 ; B)            p(o_ijk=0 | e=0 ; B)
        POe0[e] *= Po1_e[0][B] if data[o_ijk] else (1-Po1_e[0][B])
        # p(O|e=1)  p(o_ijk=1 | e=1 ; B)            p(o_ijk=0 | e=1 ; B)
        POe1[e] *= Po1_e[1][B] if data[o_ijk] else (1-Po1_e[1][B])

    for e in bm.ndindex((N,N)):

        # p(e=1,O) = p(O|e=1)*p(e=1)
        B = tuple(groups[list(e)]) # YEAH? This went from 3d->2d blocks ...
        POe0[e] *= (1-p[B]) # p(e=0)
        POe1[e] *=    p[B]  # p(e=1)

    #print "POe0: ", np.mean(POe0)
    #print "POe1: ", np.mean(POe1)
    #print 'likelihood[0]: \n', likelihood[0]
    #print 'likelihood[1]: \n', likelihood[1]
    # p(e=1|O) = p(e=1,O)/sum_e(p(e=?,O)) = p(e=1,O)/(p(e=1,O) + p(e=0,O))
    return POe1/(POe1 + POe0)

    # POe0 = np.zeros((N,N)) # p(O,e=0)
    # POe1 = np.zeros((N,N)) # p(O,e=1)
    #
    # for o_ijk in bm.ndindex(data.shape):
    #
    #     e = o_ijk[1:]
    #     B = tuple(groups[list(o_ijk)])
    #     # p(O|e=0)  p(o_ijk=1 | e=0 ; B)            p(o_ijk=0 | e=0 ; B)
    #     POe0[e] += np.log(Po1_e[0][B]) if data[o_ijk] else np.log(1-Po1_e[0][B])
    #     # p(O|e=1)  p(o_ijk=1 | e=1 ; B)            p(o_ijk=0 | e=1 ; B)
    #     POe1[e] += np.log(Po1_e[1][B]) if data[o_ijk] else np.log(1-Po1_e[1][B])
    #
    # for e in bm.ndindex((N,N)):
    #
    #     # p(e=1,O) = p(O|e=1)*p(e=1)
    #     B = tuple(groups[list(e)]) # YEAH? This went from 3d->2d blocks ...
    #     POe0[e] += np.log(1-p[B]) # p(e=0)
    #     POe1[e] += np.log(  p[B]) # p(e=1)
    #
    # # p(e=1|O) = p(e=1,O)/sum_e(p(e=?,O)) = p(e=1,O)/(p(e=1,O) + p(e=0,O))
    # #return POe1/(POe1 + POe0)
    # return 1/(np.exp(POe0 - POe1) + 1)



def count_errors(data_orig, best_guess_orig, groups_orig, k, indices=None):
    """ Counts true/false positive/negatives for each cluster-tuple.

    Takes the observations, the best_guess, and the group assignments
    """

    data = copy.copy(data_orig)
    groups = copy.copy(groups_orig)
    best_guess = copy.copy(best_guess_orig)
    # DANGER
    # DO I NEED TO COPY THE DATA?
    if indices:# and g.ndim==3:
        #print 'YES THERE ARE INDICES'
        other_indices = [i for i in range(data.shape[1]) if not i in indices]
        new_indices = np.array(indices + other_indices)
        # Reshuffles the matrices so that the first elems are the samples
        data = data[:,new_indices,:]
        data = data[:,:,new_indices]
        best_guess = best_guess[:,new_indices]
        best_guess = best_guess[new_indices,:]
        groups = groups[new_indices]
        #print new_indices

    # The structures that will hold the accuracy counts
    tp = np.zeros((k,)*3)
    fp = np.zeros((k,)*3)
    tn = np.zeros((k,)*3)
    fn = np.zeros((k,)*3)

    # Perceiver, Sender, Receiver
    for p, i, j in bm.ndindex(data.shape):
        # The Truth (we're assuming)
        t = best_guess[i,j]
        # The Observation
        o = data[p,i,j]
        # The node-group index tuple
        gix = tuple(groups[[p, i, j]])

        if       t and     o:
            tp[gix] += 1
        elif not t and     o:
            fp[gix] += 1
        elif not t and not o:
            tn[gix] += 1
        elif     t and not o:
            fn[gix] += 1
        else:
            raise Exception("Bottomed out of cases for observations")

    return tp, fp, tn, fn


def estimate_error_rates(data_orig, best_guess_orig, groups_orig, k, indices=None):

    # Now this just calls it directly -- avoids having to nest things awkwardly
    tp, fp, tn, fn = count_errors(data_orig, best_guess_orig, groups_orig, k, indices=indices)


    # # P(false_pos) = P(edge=0 | view = 1) # ... yeah? no -- no.
    # #              = c(e=0,v=1)/c(v=1)
    # #              = fp/(tp + fp)
    #
    # pfp = fp/(tp + fp)
    # pfn = fn/(tn + fn)

    # P(false_pos) = P(view = 1 | edge = 0)
    #              = c(e=0,v=1)/c(e=0)
    #              = fp/(tn + fp)

    pfp = fp/(tn + fp)
    pfn = fn/(fn + tp)

    # Div by 0 => numerator is 0 too
    pfp[np.isnan(pfp)] = 0.0
    pfn[np.isnan(pfn)] = 0.0

    # Trying to enforce an identifiability constraint ...
    # but a little arbitrarily
    #pfp[pfp >= .5] = 0.05
    #pfn[pfn >= .5] = 0.05

    return pfp, pfn


def guess(data, i, j, pfp, pfn, groups, p):

    # Preparing to build up the values:
    # P(obs | edge), P(obs | no edge)
    p1, p0 = 1, 1

    for perceiver in range(data.shape[0]):

        gix = tuple(groups[[perceiver, i, j]])
        # P(all observations | edge/no edge)
        p1 *= pfn[gix] if data[gix]==0 else 1-pfp[gix]
        p0 *= pfp[gix] if data[gix]==1 else 1-pfn[gix]

    # Multiply by the a priori possibility of the edge,
    ij = tuple(groups[[i,j]])
    # based on the blocks it would span, to finally get
    p1 *= p[ij]     # = P(edge,  obs)
    p0 *= 1 - p[ij] # = P(!edge, obs)
    # We want the posterior -- but we have all the elements.
    # pIeloI = p1/(p1+p0) # P(edge|obs) = P(edge, obs)/(P(!e,o) + P(e,o))
    # Trying to write p(e|o) only using alphanumerics. Kinda works, right?
    # return pIeloI
    return (p1 > p0) + 0 # bool => int

def accuracy(x, y):
    tn = np.sum((x+y)==0)/np.sum((1-y)**2)
    tp = np.sum((x+y)==2)/np.sum(y)
    return tn, tp


def em(data, k=2, indices=None, G_true=None, num_samples=5, iterations=20, corrected=True):

    n = data.shape[0] # The number of samples
    N = data.shape[1] # The size of the network being sampled

    G = bm.initial_condition(data)
    G_old = copy.copy(G)

    revert_indices = None
    if indices:
        other_indices = [i for i in range(data.shape[1]) if not i in indices]
        new_indices = np.array(indices + other_indices)
        # Reshuffles the matrices so that the first elems are the samples
        data = data[:,new_indices,:]
        # Because numpy is INSANE, I have to permute the dimensions 1 by 1.
        data = data[:,:,new_indices]
        # Cast so that it has list's 'index' function.
        ni = list(new_indices)
        revert_indices = np.array([ni.index(i) for i in range(data.shape[1])])
        if not G_true is None:
            G_true = G_true[new_indices,:]
            G_true = G_true[:,new_indices]
    old_groups = np.random.randint(0,2,N)
    truth, liks = bm.blockmodel(G_true, 2)
    fixed_groups = truth.groups

    # b = bm.blockmodel(data, k, iterations=iterations,
    #                         corrected=corrected,
    #                         indices=indices)

    em_iterations = 0

    est_diffs = []
    true_diffs = []
    em_lkhds = []
    accs = []
    group_diff = []

    values = []
    values.append(np.ravel(G))
    groups = []
    probs = []
    liks = []
    modliks = []
    models = []

    while True:
        sys.stdout.write("EM iteration #{} \r".format(em_iterations))
        sys.stdout.flush()
        em_iterations += 1


        b, lkhds = bm.blockmodel(G, k, iterations=iterations, corrected=corrected)
        likelihood = calculate_likelihood(data, b.groups, G, n, k)
        G = calculate_posterior(data, b.groups, likelihood, b.p, N, k)



        liks.append(likelihood)
        probs.append(np.ravel(b.p))
        modliks.append(sum([G[e]*np.log(G[e]) for e in bm.ndindex((N,N))]))
        groups.append(b.groups)
        #likelihood = calculate_likelihood(data, fixed_groups, G, n, k)#, indices)
        #G = calculate_posterior(data, fixed_groups, likelihood, b.p, N, k)#, indices)
        # Actually it's good if it treats it as UNdirected -- I'm only
        # afraid of nodes that have NO context
        Gnx = nx.from_numpy_matrix(np.round(G))#, create_using=nx.DiGraph())
        print '#CC: ', nx.number_connected_components(Gnx), " "

        est_diff =        np.sum(np.abs(G - G_old))

        group_diff.append(gd.d2(b.groups, old_groups))
        est_diffs.append(est_diff)
        true_diffs.append(np.sum(np.abs(G - G_true)))
        em_lkhds.append(b.calculate_likelihood())
        accs.append(accuracy(np.round(G), G_true))
        #groups.append(b.groups)
        old_groups = copy.copy(b.groups)
        values.append(np.ravel(G))
        models.append(b)

        if est_diff < 1e-1 or em_iterations > 100:

            b, lkhds = bm.blockmodel(np.round(G), k,
                                    iterations=iterations,
                                    corrected=corrected)#,
                                    #indices=indices)

            if not revert_indices is None:
                b.groups = b.groups[revert_indices]
                b.g = b.g[revert_indices,:]
                b.g = b.g[:,revert_indices]
                G = G[revert_indices,:]
                G = G[:,revert_indices]

            return np.round(G), b, est_diffs, true_diffs, em_lkhds, accs, group_diff, values, groups, probs, liks, modliks, models
        else:
            G_old = copy.copy(G)





def test_count_errors(n, k, e):

    # em.test_count_errors(20, 2, np.array([[1,2],[3,4]]))

    # I want to see if ... fp + fn == e
    # first ... necessary?
    e = copy.copy(e)
    # 1) Make a PERFECT model,
    x = bm.blocked_matrix(n, k)
    parts = bm.partition(n, k)
    # [[1,2,3],[4,5,6]] => [0,0,0,1,1,1 ...]
    groups = np.concatenate([[i]*len(parts[i]) for i in range(k)])
    # 2) Copy it, for the perceptions
    y = copy.copy(x)
    # Have two people, with the same views ... yes, counts double.
    #x = np.array(x, ndmin=3) # So the indexing over 'perceivers' works right
    x = np.array([x, x]) # in case I want several observers
    # 3) Turn the NUMBERS of errors into actual coordinates
    edges = list(bm.ndindex((n,)*e.ndim))
    eix = np.zeros(y.shape, dtype=np.bool)
    for edge in edges:
        # The groups of the nodes involved in the edge ('Group IndeX')
        gix = tuple(groups[list(edge)])
        # If there are errors left to disperse from block gix,
        if e[gix]:
            # Mark the index for the error
            eix[edge] = True
            # And reduce the number of remaining errors to distribute
            e[gix] -= 1
    # 4) Flip the bits on the error spots
    y[eix] = ((1-y)**2)[eix]

    print 'y: \n', np.int64(y)

    # 4) See if count errors' recovers them.
    # data, best_guess, groups, k
    out = count_errors(x, y, groups, k)

    return out

def test_estimate_error_rates():
    pass

def test_guess():
    pass

def faulty_observations(indices=None, N=20, simple=True):
    # CONCOCT: ... some cases for the likelihood/posterior TO WHICH I know
    # the answers. So ... Ah -- so -- DRAW from them. Directly. Right.
    # I want ... p(e|o) AND p(o|e) ... etc. Right. And for NOW, have it be ONE
    # group. So -- p(e=1) = bm.blocked_matrix(20, 2, on=1, off=.1)
    # p(o=1|e=1) = .9 #, .8 for groups 1, 2.  (NO -)

    if not indices:
        indices = range(0,20,2)

    truth = bm.blocked_matrix(N, 2, on=.9, off=.1) # => ~.8 .1 b/c diagonal
    groups = np.zeros(N)
    groups[10:] = 1

    pfp  = np.array([ [ [.1, .2],
                        [.2, .1] ],

                      [ [.2, .1],
                        [.1, .2] ]
                    ])
    pfn  = np.array([ [ [.0, .1],
                        [.1, .0] ],

                      [ [.1, .0],
                        [.0, .1] ]
                    ])

    all_obs = []
    # generate data
    for i in indices:#range(10):
        obs = copy.copy(truth)
        if simple:
            obs = bm.blocked_matrix(N,2)
        else:
            for ix in bm.ndindex((N,N)):
                B = tuple(groups[[i, ix[0], ix[1]]])
                if obs[ix]:
                    if np.random.random() < pfn[B]:
                        obs[ix] = 0
                else:
                    if np.random.random() < pfp[B]:
                        obs[ix] = 1
        all_obs.append(obs)

    return np.array(all_obs), pfp, pfn, truth, groups

def permute(x, indices):
    if x.ndim==2:
        x = x[indices,:]
        x = x[:,indices]
    elif x.ndim==3:
        # NOT the first index
        x = x[:,indices,:]
        x = x[:,:,indices]
    return x

def test_posterior():
    pass

if __name__ == "__main__":

    # Putting it down here is particularly important because it doesn't play
    # at all with pypy, so now the OTHER EM functions can be imported
    # (via the script) without throwing that wrench.
    #from matplotlib import pyplot as plt



    # all data, idem, consensus, idem, self-reported, idem
    advice, friendship, ca, cf, sa, sf = bm.read_css()
    # print 'yeah? ', np.all([np.all(cf==X[i]) for i in range(len(X))])
    # raw_input()
    # G_hat, b =  em(X, np.random.randint(2,size=(21,21)), k=2, num_samples=5, iterations=20, corrected=True)

    Y, pfp, pfn, truth, y_groups = faulty_observations(indices=range(20), simple=False)
    likelihood = calculate_likelihood(Y, y_groups, truth, 20, 2)
    p = np.array([[.8,.1],[.1,.8]])
    posterior  = calculate_posterior(Y, y_groups, likelihood, p, 20, 2)
    #print 'posterior: \n', np.int64(np.round(posterior))
    #sys.exit()


    X_true = bm.blocked_matrix(20, 2, on=.9, off=.1)
    X  = [bm.blocked_matrix(20, 2, on=1., off=0.) for i in range(10)]
    X += [bm.blocked_matrix(20, 2, on=1., off=0.) for i in range(10)]
    X = np.array(X)
    indices = range(0,20,1)
    new_indices = indices + [i for i in range(20) if not i in indices]

    Y, pfp, pfn, truth, y_groups = faulty_observations(indices)

    groups = np.zeros(20)
    groups[10:] = 1
    # p(o=1|e=0/1), i.e. pfp, ptp
    lkhd = calculate_likelihood(permute(Y,     new_indices),
                                permute(groups,new_indices),
                                permute(truth, new_indices), len(groups), 2)
    #sys.exit()


    indices = range(0,18,1)
    connected = [i for i in range(21) if not i in (8, 9, 19)]
    f_con = friendship[connected,:,:]
    f_con =      f_con[:,connected,:]
    f_con =      f_con[:,:,connected]
    cf_con =        cf[connected,:]
    cf_con =    cf_con[:,connected]
    cfb, liks = bm.blockmodel(cf_con, 2, iterations=100)
    cab, liks = bm.blockmodel(ca, 2, iterations=100)

    #cfb.plot()


    # pfp_star, pfn_star = estimate_error_rates(f_con[indices], cf_con, cfb.groups, 2, indices=indices)
    # graph, model, est_diffs, true_diffs, em_lkhds, accs, group_diffs, values, est_groups, probs, liks, modliks, models = em(f_con[indices], k=2, G_true=cf_con, iterations=100, indices=indices, corrected=False)#X, G_true=bm.blocked_matrix(20,2))#, iterations=100)
    # pfp_hat, pfn_hat = estimate_error_rates(f_con[indices], graph, est_groups[-1], 2, indices=indices)

    pfp_star, pfn_star = estimate_error_rates(advice[indices], ca, cab.groups, 2, indices=indices)
    #graph, model, est_diffs, true_diffs, em_lkhds, accs, group_diffs, values, est_groups, probs, liks, modliks, models 
    stuff = em(advice[indices], k=2, G_true=ca, iterations=100, indices=indices)#X, G_true=bm.blocked_matrix(20,2))#, iterations=100)
    pfp_hat, pfn_hat = estimate_error_rates(advice[indices], graph, est_groups[-1], 2, indices=indices)

    # pfp_star, pfn_star = estimate_error_rates(Y, truth, y_groups, 2, indices=indices)
    # graph, model, est_diffs, true_diffs, em_lkhds, accs, group_diffs, values, est_groups, probs, liks = em(Y[indices], G_true=truth, indices=indices, iterations=100)#, iterations=100)
    # pfp_hat, pfn_hat = estimate_error_rates(Y, truth, est_groups[-1], 2, indices=indices)



    #graph, model, est_diffs, true_diffs, em_lkhds, accs, group_diffs, values, groups, probs = em(X, G_true=bm.blocked_matrix(20,2), indices=indices)#, iterations=100)




    # NOW AGAIN, for advice ... but also KEEP EVERYTHING
    # accs = []
    # bcf = bm.blockmodel(cf, k=2, corrected=True)
    # for num in range(21):
    #     print 'num: ', num
    #     G_hat, b, nits = em(friendship[range(num)], k=2, iterations=100, corrected=True)
    #     accs.append(accuracy(np.round(G_hat), cf))
    #     print 'accuracy after: ', accuracy(np.round(G_hat), cf)
    #     print 'bG.p: \n', b.p
    #     print 'bcf.p: \n', bcf.p
    #
    # plt.plot([i[0] for i in accs])
    # plt.plot([i[1] for i in accs])
    # plt.plot([np.prod(i) for i in accs])
    # plt.ylim([-.1,1.1])
    # plt.show()
    #
    # vals = []
    # for e in bm.ndindex((18,18)):
    #     if cf_con[e]:
    #         vals.append(np.mean(f_con[:,e[0],e[1]]))
    # plt.hist(vals)
    # plt.show()

    # THIS DOESN"T WORK PARTICULARLY WELL
    # print 'distance: ', np.sum(np.abs(G_hat-cf))
    # for t in np.linspace(0,1, 10):
    #     print 't: ', t, ' distance: ', np.sum(np.abs((G_hat > t)+0 - cf))


    # data = {'a': [advice, ca], 'f': [friendship, cf]}
    # # all_data, consensus, k, num_samples, iterations, corrected
    # defaults = ('f', 2, 10, 20, 1)
    #
    # v = sys.argv[1:]
    # args = [v[i] if len(v) > i else defaults[i] for i in range(len(defaults))]
    # #print 'args: ', args
    # args = data[args[0]] + list(np.int64(args[1:])) # array cast [str]->[int]
    #
    # # best_guess, model, pfp,  pfn =
    # # em(friendship, cf, k=3, num_samples=10, iterations=20)
    # best_guess, b, pfp, pfn = em(*args)
    #
    # print 'final accuracy:          {:.3f}'.format(np.sum(best_guess == args[1])/441.)
    # print 'pfp: \n', pfp
    # print 'pfn: \n', pfn
    # print 'b.p: \n', b.p
    # print 'b.m: \n', b.m
    # print 'b.groups: ', b.groups
    # b.plot()
