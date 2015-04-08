
import blockmodels as bm
import css_em as em

a, f, ca, cf, sa, sf = bm.read_css()

# cf_indices = []
# cf_20_diffs = []
# for i in range(21):
#     bcf, lkhds = bm.blockmodel(f[i], 2)
#     # The max of the first 20 trials --
#     max20 = max(lkhds[:20])
#     # Where is the max of all 100 trials?
#     cf_indices.append(lkhds.index(bcf.calculate_likelihood()))
#     # How far away was max20 from maxtotal?
#     cf_20_diffs.append(abs(max20 - bcf.calculate_likelihood()))
#     print i
#
#
# ca_indices = []
# ca_20_diffs = []
# for i in range(21):
#     bca, lkhds = bm.blockmodel(a[i], 2)
#     # The max of the first 20 trials --
#     max20 = max(lkhds[:20])
#     # Where is the max of all 100 trials?
#     ca_indices.append(lkhds.index(bca.calculate_likelihood()))
#     # How far away was max20 from maxtotal?
#     ca_20_diffs.append(abs(max20 - bca.calculate_likelihood()))
#     print i


graph, model, est_diffs, true_diffs, em_lkhds = em.em(f, G_true=cf, iterations=100)
#plt.plot(diffs)
#plt.show()

# #
# f_indices = []
# a_indices = []
#
# for i in range(100):
#     bf, lkhds = bm.blockmodel(f, 2)
#     f_indices.append(lkhds.index(bf.calculate_likelihood()))
#     print i
#
# for i in range(100):
#     ba, lkhds = bm.blockmodel(a, 2)
#     a_indices.append(lkhds.index(ba.calculate_likelihood()))
#     print i
