import pandas as pd
import glob
import numpy as np


filenames = glob.glob('ZVCV_comparison/*.csv')
cnt = 0
cnt1 = 0
mcmc_var_ = []
linear_var_ = []
quad_var_ = []
linear_improve = []
quad_improve = []

for i, filename in enumerate(filenames):
    print(str(i) + "  filename: " + filename)
    flag_ = True
    result = pd.read_csv(filename)
    mcmc_var = np.array(result['mcmc_samples_var'])
    linear_re = np.array(result['CV_linear_mean'])
    linear_var = np.array(result['CV_linear_var'])
    quad_re = np.array(result['CV_quad_mean'])
    quad_var = np.array(result['CV_quad_var'])

    mcmc_var_.append(np.mean(mcmc_var))
    if np.all(linear_re == -1.0):
        linear_improve.append(np.nan)
        linear_var_.append(np.nan)
    else:
        if np.any(linear_var == 0):
            linear_improve_ = np.inf
        else:
            linear_improve_ = np.log(np.mean(mcmc_var / linear_var))

        linear_improve.append(linear_improve_)
        linear_var_.append(np.mean(linear_var))

    if np.all(quad_re == -1.0) or np.all(quad_re == 0.0):
        quad_improve.append(np.nan)
        quad_var_.append(np.nan)
    else:
        if np.any(quad_var == 0):
            quad_improve_ = np.inf
        else:
            quad_improve_ = np.log(np.mean(mcmc_var / quad_var))
        quad_improve.append(quad_improve_)
        quad_var_.append(np.mean(quad_var))

    print("---------------------------------")

results_df = pd.DataFrame({'Examples': filenames, 'linear CV improvement (log)': linear_improve,
                           'quadratic CV improvement (log)': quad_improve, 'pystan var': mcmc_var_,
                           'linear CV average variance': linear_var_, 'quadratic CV average variance': quad_var_})
results_df.to_csv('conclusions.csv')
