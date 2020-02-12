from PerformanceTest_ZVCV import run_ZVCV
import pandas as pd
import numpy as np
import sys


# Parameters
output_path = 'performance-tests-cmdstan/ZVCV_comparison/'

# Read list of examples
list_examples = []
with open('list_of_examples.txt', 'r') as f:
    example_path = f.readline()
    while example_path:
        example_path = example_path.strip()
        list_examples.append(example_path)
        example_path = f.readline()
print(list_examples)

# run test
start_i = 435    # 434
for i, example_path in enumerate(list_examples):
    if i < start_i:
        continue
    print('------------------------------------------------------------------------')
    print(i)
    print(example_path)
    print()
    sys.stderr.write('------------------------------------------------------------------------')
    print(i)
    print(example_path)
    print()
    # example_name_parts = example_path.split('/')
    example_name = example_path.replace('/', '--')
    unconstrain_mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples = run_ZVCV(example_path)

    if unconstrain_mcmc_samples is not None:
        unconstrain_mcmc_mean = np.mean(unconstrain_mcmc_samples, axis=0)
        unconstrain_mcmc_var = np.var(unconstrain_mcmc_samples, axis=0)
        if cv_linear_mcmc_samples is not None:
            cv_linear_mcmc_mean = np.mean(cv_linear_mcmc_samples, axis=0)
            cv_linear_mcmc_var = np.var(cv_linear_mcmc_samples, axis=0)
        else:
            cv_linear_mcmc_mean = np.ones_like(unconstrain_mcmc_mean) * -1
            cv_linear_mcmc_var = np.ones_like(unconstrain_mcmc_var) * -1

        if cv_quad_mcmc_samples is not None:
            cv_quad_mcmc_mean = np.mean(cv_quad_mcmc_samples, axis=0)
            cv_quad_mcmc_var = np.var(cv_quad_mcmc_samples, axis=0)
        else:
            cv_quad_mcmc_mean = np.ones_like(unconstrain_mcmc_mean) * -1
            cv_quad_mcmc_var = np.ones_like(unconstrain_mcmc_var) * -1

        result_dic = {'mcmc_samples_mean': unconstrain_mcmc_mean, 'CV_linear_mean': cv_linear_mcmc_mean, 'CV_quad_mean': cv_quad_mcmc_mean,
                      'mcmc_samples_var': unconstrain_mcmc_var, 'CV_linear_var': cv_linear_mcmc_var, 'CV_quad_var': cv_quad_mcmc_var}

        result_df = pd.DataFrame(result_dic)
        result_df.to_csv(output_path + example_name + '.csv')

    print('------------------------------------------------------------------------')

    if 0:
        with open(output_path + example_name + '.dat', 'w') as f:
            f.write('MCMC without ZVCV: \n')
            f.write(str(unconstrain_mcmc_samples))
            f.write('------------------------------------------ \n')
            f.write('------------------------------------------ \n')
            f.write('MCMC with linear ZVCV: \n')
            f.write(str(cv_linear_mcmc_samples))
            f.write('------------------------------------------ \n')
            f.write('------------------------------------------ \n')
            f.write('MCMC without quadratic ZVCV: \n')
            f.write(str(cv_quad_mcmc_samples))
            f.write('------------------------------------------ \n')
            f.write('------------------------------------------ \n')

