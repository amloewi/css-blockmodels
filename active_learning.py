
import datetime
#import pickle
import json
import numpy as np

#from r import *
# Necessary?
#library('kernlab')

import blockmodels as bm
import css_em as em
import grouping_distance as gd

###############
#


def svm(data):
    pass_to_R(data, "data")
    pass_to_R()
    # And, fuck -- in/out sample acc?
    ###################################
    # model.formula = formula(train$sa ~ data.matrix(train[,-c(1:6)]))
    # train.svm =   ksvm(model.formula, type="C-svc")
    # out.sample = predict(train.svm, data.matrix(test[,-c(1:6)]))

    R.r("model.formula = formula(data[,1] ~ data[,-1])")
    R.r("m = ksvm(model.formula, type='C-svc')")
    R.r("")
    R.r("")
    # Gotta reshape, post prediction


def familiar(data, sample, N, rule):

    counts = np.zeros(N)
    f = max if rule=="most" else min
    for ego in sample:
        for alter in range(N):
            if data[ego, ego, alter] or data[ego, alter, ego]:
                counts[alter] += 1
    # Returns all the indices of args that are the max/min value
    options = [counts[i] for i in range(N) if not i in sample]
    # This filtering is awkward, but ... I can't think of anything much
    # cleaner right now.
    all_matches = np.arange(N)[counts==f(options)]# & ~np.in1d(range(N), sample)]
    # This is why this still needs to be run more than once.
    matches = [m for m in all_matches if not m in sample]
    next_person = np.random.choice(matches)

    return next_person


def active(data, sample, N, rule):

    # Flatten to a 2D matrix
    votes = np.sum(data[sample], axis=0)
    # For 2D, 0 is columns (squash from above). Is that IN though?
    axis = 0 if rule=="in" else 1
    counts = np.sum(votes, axis=axis)
    # Returns all the indices of args that are the max/min value
    options = [counts[i] for i in range(N) if not i in sample]
    all_matches = np.arange(N)[counts==max(options)]# & ~np.in1d(range(N), sample)]
    # This is why this still needs to be run more than once.
    #next_person = np.random.choose(all_matches)
    matches = [m for m in all_matches if not m in sample]
    next_person = np.random.choice(matches)


    return next_person



def kaboom(data, consensus, file, iterations=1, rule="familiar"):
    """Does rule-based sample addition (active learning) based on ties.

    Returns lists of group similarity, of tpr/tnr tuples using
    the all-data groups, and consensus graphs as ground truth, as well as
    other models

    In other words, rather than choosing the next sample randomly as you
    increase N, chooses it based on a 'rule' such as 'who does the current
    sample say is the most popular person in the network.'

    'Rules' include familiar, unfamiliar, popular, and gregarious -- these
    mean the person who has the most MOST relationships WITH the current sample,
    who has the FEWEST, who the sample says RECEIVES the most ties, and who the
    sample says SENDS the most ties. (And 'random, which should be obvious',
    and 'linear' which runs the samples in 'order', for stability.)
    """

    print 'Started ', file, ' at ', datetime.datetime.now()

    N = data.shape[1]
    # THIS IS THE FULL_DATA MODEL -- as opposed to the consensus one.
    # Which do I actually want?
    b = bm.blockmodel(consensus, 2, corrected=True, iterations=100)

    all_iterations = []
    for it in range(iterations):
        sample = []
        one_iteration = []
        for n in range(1,21+1):
            one_sample = {}
            #print 'On n= ', n
            #sample = people[0:n]
            if   rule=="familiar":
                next_person = familiar(data, sample, N, 'most')
            elif rule=="unfamiliar":
                next_person = familiar(data, sample, N, 'least')
            elif rule=="popular":
                next_person = active(data, sample, N, "in")
            elif rule=="gregarious":
                next_person = active(data, sample, N, "out")
            elif rule=="linear":
                next_person = n-1
            elif rule=="random":
                left = [i for i in range(N) if not i in sample]
                next_person = np.random.choice(left)
            else:
                raise ValueError("Your 'rule' did not match the options.")

            sample.append(next_person)

            # Meaning, again, this is doing data-modeling, instead of
            # on the inferred 2d model --- is that right?
            # But HERE, I have to decide how I'm going to classify.
            G_hat, b_hat, nits = em.em(data[sample], k=2, indices=sample)

            # Now, a partial 3d modeling for comparison ...
            m = bm.blockmodel(data[sample], 2, indices=sample)

            # NOW MAKE AND COLLECT ALL THE DATA!
            inferred_edge_accuracy = em.accuracy(G_hat >= .5, consensus)
            full_data_groups_distance        = gd.d2(b.groups, m.groups)
            inferred_network_groups_distance = gd.d2(b.groups, b_hat.groups)

            # What am I trying to estimate here?
            # I DO want to treat consensus as ground truth,
            # but what are the ground truth groups? The data run on consensus,
            # or on ALL data? Gotta do both I guess.
            four_b = em.count_errors(data[sample], consensus, b.groups, 2, indices=sample)
            pfp_b, pfn_b = em.estimate_error_rates(*four_b)
            four_m = em.count_errors(data[sample], consensus, m.groups, 2, indices=sample)
            pfp_m, pfn_m = em.estimate_error_rates(*four_m)

            # And just stick 'em all in the list. (Casting, first, to
            # traditional python lists, because otherwise they're PYPY arrays,
            # not even NUMpy arrays, and so can't be used in the same script as
            # matplotlib -- which is the whole point of collecting this data.)
            one_sample['sample'] = sample
            one_sample['edge_accuracy'] = inferred_edge_accuracy
            # These are already floats, but PYPY floats -- not Cpython. Trust me.
            one_sample['full_groups_accuracy'] = float(full_data_groups_distance)
            one_sample['g_hat_groups_accuracy'] = float(inferred_network_groups_distance)
            one_sample['em_iterations'] = nits

            one_sample['b_hat_p'] = b_hat.p.tolist()
            one_sample['full_data_p'] = m.p.tolist()

            one_sample['b_groups_pfp'] = pfp_b.tolist()
            one_sample['b_groups_pfn'] = pfn_b.tolist()

            one_sample['full_groups_pfp'] = pfp_m.tolist()
            one_sample['full_groups_pfn'] = pfn_m.tolist()

            one_iteration.append(one_sample)

        all_iterations.append(one_iteration)

    print 'Finished one KABOOM'
    print 'Finished ', file, ' at ', datetime.datetime.now()

    f = open(file, 'wb')
    #pickle.dump(all_iterations, f)
    json.dump(all_iterations, f)
    f.close()

    #return all_runs #groups_all_runs, accuracy_all_runs



if __name__ == "__main__":

    a, f, ca, cf, sa, sf = bm.read_css()

    # data, consensus, iterations, rule

    # advice_groups_linear, advice_accuracy_linear = kaboom(a, ca, 1, "linear")
    advice_groups_rand,  advice_accuracy_rand    = kaboom(a, ca, 10, "random")
    advice_groups_fam,   advice_accuracy_fam     = kaboom(a, ca, 10, "familiar")
    advice_groups_unfam, advice_accuracy_unfam   = kaboom(a, ca, 10, "unfamiliar")
    advice_groups_pop,   advice_accuracy_pop     = kaboom(a, ca, 10, "popular")
    advice_groups_greg,  advice_accuracy_greg    = kaboom(a, ca, 10, "gregarious")


    friends_groups_rand,  friends_accuracy_rand  = kaboom(f, cf, 10, "random")
    friends_groups_fam,   friends_accuracy_fam   = kaboom(f, cf, 10, "familiar")
    friends_groups_unfam, friends_accuracy_unfam = kaboom(f, cf, 10, "unfamiliar")
    friends_groups_pop,   friends_accuracy_pop   = kaboom(f, cf, 10, "popular")
    friends_groups_greg,  friends_accuracy_greg  = kaboom(f, cf, 10, "gregarious")
