import pandas as pd
import glob
import numpy as np


filenames = glob.glob('ZVCV_comparison/*.csv')
cnt = 0
cnt1 = 0
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
        flag_ = False
        print("linear CV error...")

    if np.all(quad_re == -1.0):
        flag_ = False
        print("quadratic CV error...")
    if np.all(quad_re == 0.0):
        flag_ = False
        print("quadratic CV is not processed due to potential memory issue...")

    if flag_:
        cnt += 1

    if flag_:
        mcmc_var_mean = np.mean(mcmc_var)
        linear_var_mean = np.mean(linear_var)
        quad_var_mean = np.mean(quad_var)

        if quad_var_mean < mcmc_var_mean:
            cnt1 += 1

    print("---------------------------------")
