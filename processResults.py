import pandas as pd
import glob
import numpy as np


filenames = glob.glob('ZVCV_comparison/*.csv')
cnt = 0
cnt1 = 0
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
    '''
    if np.all(mcmc_var == 0.0):
        flag_ = False
        print("mcmc error 1...")
    if np.all(mcmc_var == -1.0):
        flag_ = False
        print("mcmc error 2...")
    '''
    if np.all(linear_re == -1.0):
        linear_improve_ = np.nan
        linear_improve.append(linear_improve_)
    else:
        linear_improve_ = np.log(np.mean(mcmc_var / linear_var))
        linear_improve.append(linear_improve_)
    if np.all(quad_re == -1.0) or np.all(quad_re == 0.0):
        quad_improve_ = np.nan
        quad_improve.append(quad_improve_)
    else:
        quad_improve_ = np.log(np.mean(mcmc_var / quad_var))
        quad_improve.append(quad_improve_)

    print("---------------------------------")

results_df = pd.DataFrame({'Examples': filenames, 'linear CV improvement': linear_improve,
                           'quadratic CV improvement': quad_improve})
results_df.to_csv('conclusions.csv')
