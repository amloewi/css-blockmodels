

# import numpy as np
# import pandas as pd
#
# # base = library('base') -- import packages from R
# from rpy2.robjects.packages import importr as library
# # R.R('x <- 1') AND R.Array('...') etc -- the core interface
# import rpy2.robjects as R
# # Not clear what this does yet, but allows numpy->R easily?
# import rpy2.robjects.numpy2ri
# # Guess if I want to use formulas, I do really need pandas though --
# # rdf = pd2r.convert_to_r_dataframe(pdf)
# import pandas.rpy.common as pd2r
#
#
# def setup_R():
#
#     import numpy as np
#     import pandas as pd
#
#     # base = library('base') -- import packages from R
#     from rpy2.robjects.packages import importr as library
#     # R.R('x <- 1') AND R.Array('...') etc -- the core interface
#     import rpy2.robjects as R
#     # Not clear what this does yet, but allows numpy->R easily?
#     import rpy2.robjects.numpy2ri
#     # Guess if I want to use formulas, I do really need pandas though --
#     # rdf = pd2r.convert_to_r_dataframe(pdf)
#     import pandas.rpy.common as pd2r
#
#
#
# def to_rdf(df, name):
#     converted = pd2r.convert_to_r_dataframe(df)
#     R.globalenv[name] = converted
#     return converted


from r import *

if __name__ == '__main__':

    base    = library('base')
    stats   = library('stats')
    gam     = library('gam')
    kernlab = library('kernlab')

    x = np.random.randn(100)
    df = pd.DataFrame({'y':2*x+1, 'x':x})


    rdf = dataframe(df, 'rdf')
    xgam = R.r("gam(y ~ x, family=gaussian, data=rdf)")

    print base.summary(xgam)
