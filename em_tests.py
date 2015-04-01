
import numpy as np

import blockmodels as bm
import css_em as em

# Do it
# 1) With k > 2
# 2) Different rates for diffent groups
#
G_true = bm.blocked_matrix(20, 2, dim=2) # on=1, off=0
X  = [bm.blocked_matrix(20, 2, dim=2, on=.8, off=.2) for i in range(5)]
X += [bm.blocked_matrix(20, 2, dim=2, on=.9, off=.1) for i in range(5)]
# Concatenate the lists -- adding arrays adds the elements.
X = np.array(X)


# HAVE to ROUND the final estimate. (Otherwise -- weird shit.)
# Find a GOOD STARTING POINT.
b = bm.blockmodel(G_true, k=2)

#
# INDICES: it thinks the ten samples are the first ten people, and because
# they're blocked in order, they all end up in the same cluster -- meaning
# there just objectively isn't anything for the other group. SO.
#
indices = range(0,20,2)
G_hat, b_hat = em.em(X, k=2, indices=indices)

# Maybe it's the GROUPS -- there isn't a ground truth ...
# But it also shouldn't matter. Wait -- of course it does --
# Why aren't they getting the groups right?

# THIS isn't taking indices into account. Duh. Asshole. (Yeah?)
pfp, pfn = em.estimate_error_rates(*em.count_errors(X, G_true, b.groups, 2, indices))

# What ARE all these things? What SHOULD I be seeing?


print 'b.p: \n', b.p
print 'b_hat.p: \n', b_hat.p
print 'pfp: \n ', pfp
print 'pfn: \n ', pfn
