
import numpy as np

def mean_hamming(x, y):
  """Mean Hamming ... similarity? between vectors x, y.


  """
  return np.mean(x==y)

def d1(x, y):
    """The max mean Hamming similarity between vectors x, y
    """

    same = mean_hamming(x, y)
    flipped = mean_hamming(x, (1-y)**2)
    return max([same, flipped])


def d2(x, y):

    mx = np.zeros((len(x), len(x)))
    my = np.zeros((len(x), len(x)))

    for i, x1 in enumerate(x):
        for j, x2 in enumerate(x):
            # Are they in the same group?
            mx[i,j] = x1==x2


    for i, y1 in enumerate(y):
        for j, y2 in enumerate(y):
            # Are they in the same group?
            my[i,j] = y1==y2

    return mean_hamming(mx, my)


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import blockmodels as bm

    a, f, ca, cf, sa, sf = bm.read_css()

    comps = []
    gt = bm.blockmodel(f, 2)
    for i in range(1,21+1):
        b = bm.blockmodel(f[range(i)], 2)
        comps.append(d2(gt.groups, b.groups))
    plt.plot(comps)
    bm.show()
