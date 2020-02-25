import numpy as np
import rpy2.robjects as robjects
import re
from PyStan_control_variate.ControlVariate.control_variate import control_variate_linear, control_variate_quadratic
from PyStan_control_variate.ControlVariate.plot_comparison import plot_comparison
import pystan
from os import path
import sys
import pandas as pd
import argparse
from os.path import exists


def verifyDataType(model, data):
    model_str = model.model_code
    data_patch = re.search('data[ ]*{([^{}]*)}', model_str)
    data_str = data_patch.group(1)
    data_lines = data_str.split('\n')
    var_type_dic = {}
    for line in data_lines:
        valid_line = re.search('(.*);', line)
        if valid_line:
            valid_line = valid_line.group(1)

            # pull out size
            size_part = re.search('\[([^\[\]]*)\]', valid_line)
            if size_part is not None:
                valid_line_1 = re.sub('\[([^\[\]]*)\]', '', valid_line)
            else:
                valid_line_1 = valid_line
            range_str = re.search('\<([^\<\>]*)\>', valid_line_1)
            if range_str is not None:
                valid_line_2 = re.sub('\<([^\<\>]*)\>', '', valid_line)
            else:
                valid_line_2 = valid_line_1
            sep_line = re.search('[ ]*([^ ]*)[ ]*([^ \[\]]*)', valid_line_2)
            if sep_line:
                type_str = sep_line.group(1)
                var_str = sep_line.group(2)
                var_type_dic[var_str] = type_str
    for data_key in data:
        if data_key in var_type_dic:
            claimed_type = var_type_dic[data_key]
            if claimed_type.startswith('int'):
                data[data_key] = np.int32(data[data_key])
        else:
            print("verify data type failed!" + data_key + " is not in the stan-model file...")

    return data


def getParameterNames(model):
    model_str = model.model_code
    model_str = re.sub('[#|//].*', '', model_str)
    model_str = model_str.strip()
    data_patch = re.search('parameters[ ]*{([^{}]*)}', model_str)
    data_str = data_patch.group(1)
    data_lines = data_str.split('\n')
    var_type_dic = {}
    parameter_names = []
    for line in data_lines:
        line = re.sub('(//.*)', '', line)
        valid_line = re.search('(.*);', line)
        if valid_line:
            valid_line = valid_line.group(1)
            # pull out size in []
            size_part = re.search('\[([^\[\]]*)\]', valid_line)
            if size_part is not None:
                valid_line = re.sub('\[([^\[\]]*)\]', '', valid_line)
            range_str = re.search('\<([^\<\>]*)\>', valid_line)
            if range_str is not None:
                valid_line = re.sub('\<([^\<\>]*)\>', '', valid_line)
            sep_line = re.search('[ ]*([^ ]*)[ ]*([^ \[\]]*)', valid_line)
            if sep_line:
                type_str = sep_line.group(1)
                var_str = sep_line.group(2)
                var_type_dic[var_str] = type_str
                parameter_names.append(var_str)

    return parameter_names


def run_ZVCV(file_dir, output_dir):
    output_filename = file_dir.replace('/', '@')
    if exists(output_dir + output_filename + '.csv'):
        return 0

    print('Example Path: ' + file_dir)
    sys.stderr.write('Example Path: ' + file_dir + ' \n')
    unconstrain_mcmc_samples = None
    cv_linear_mcmc_samples = None
    cv_quad_mcmc_samples = None
    # file_dir = 'example-models/ARM/Ch.17/latent_glm_17.7'
    try:
        robjects.globalenv.clear()
        # Assume the stan model file ends with .stan
        model_file = file_dir + '.stan'
        # Assume the data file ends with .data.R
        data_file = file_dir + '.data.R'
        # read data into env
        if path.exists(data_file):
            robjects.r['source'](data_file)
            # variables
            vars = list(robjects.globalenv.keys())
            if len(vars) > 0:
                data = {}
                for var in vars:
                    data_ = np.array(robjects.globalenv.find(var))
                    if (data_.ndim == 1) and (data_.shape[0] == 1):
                        data[var] = data_[0]
                    else:
                        data[var] = data_
            else:
                data = None
        else:
            data = None

        # run stan
        sm = pystan.StanModel(file=model_file)
        parameter_names = getParameterNames(sm)

        if data is not None:
            data = verifyDataType(sm, data)
        fit = sm.sampling(data=data, chains=1, iter=5000, verbose=True)

        # Extract parameters
        parameter_extract = fit.extract()
        num_of_iter = parameter_extract['lp__'].shape[0]

        # Unconstraint mcmc samples.
        unconstrain_mcmc_samples = []
        for i in range(num_of_iter):
            tmp_dict = {}
            for j, parameter_name in enumerate(parameter_names):
                tmp_dict[parameter_name] = parameter_extract[parameter_name][i]
            unconstrain_mcmc_samples.append(fit.unconstrain_pars(tmp_dict))
        unconstrain_mcmc_samples = np.asarray(unconstrain_mcmc_samples)

        # Calculate gradients of the log-probability
        # In this case, it seems unconstraint and constraint parameters are the same.
        grad_log_prob_val = []
        for i in range(num_of_iter):
            grad_log_prob_val.append(fit.grad_log_prob(unconstrain_mcmc_samples[i], adjust_transform=False))
        grad_log_prob_val = np.asarray(grad_log_prob_val)

        # Run control variates
        cv_linear_mcmc_samples = control_variate_linear(unconstrain_mcmc_samples, grad_log_prob_val)
        if unconstrain_mcmc_samples.shape[1] <= 100:
            cv_quad_mcmc_samples = control_variate_quadratic(unconstrain_mcmc_samples, grad_log_prob_val)
        else:
            cv_quad_mcmc_samples = np.zeros_like(cv_linear_mcmc_samples)
        # plot_comparison(unconstrain_mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples, fig_name='normal.png', fig_path='/home/ubuntu/', fig_size=(8, 8))

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

            if unconstrain_mcmc_mean.size == 1:
                result_dic = {'mcmc_samples_mean': [unconstrain_mcmc_mean], 'CV_linear_mean': [cv_linear_mcmc_mean],
                              'CV_quad_mean': [cv_quad_mcmc_mean],
                              'mcmc_samples_var': [unconstrain_mcmc_var], 'CV_linear_var': [cv_linear_mcmc_var],
                              'CV_quad_var': [cv_quad_mcmc_var]}
            else:
                result_dic = {'mcmc_samples_mean': unconstrain_mcmc_mean, 'CV_linear_mean': cv_linear_mcmc_mean,
                              'CV_quad_mean': cv_quad_mcmc_mean,
                              'mcmc_samples_var': unconstrain_mcmc_var, 'CV_linear_var': cv_linear_mcmc_var,
                              'CV_quad_var': cv_quad_mcmc_var}


            result_df = pd.DataFrame(result_dic)
            result_df.to_csv(output_dir + output_filename + '.csv')

        print('------------------------------------------------------------------------')
        sys.stderr.write('------------------------------------------------------------------------ \n')

    except Exception as e:
        print(e)

    return unconstrain_mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples


if __name__ == "__main__":
    # example: python PerformanceTest_ZVCV.py "example-models/BPA/Ch.07/cjs_mnl_ran/" "ZVCV-results-1/"
    parser = argparse.ArgumentParser(description='Input example path and output path...')
    parser.add_argument('file_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    run_ZVCV(args.file_dir, args.output_dir)


