import pystan
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import re
from PyStan_control_variate.ControlVariate.control_variate import control_variate_linear, control_variate_quadratic
from PyStan_control_variate.ControlVariate.plot_comparison import plot_comparison


def verifyDataType(model, data):
    model_str = model.model_code
    data_patch = re.search('data[ ]*{([^{|^}]*)}', model_str)
    data_str = data_patch.group(1)
    data_lines = data_str.split('\n')
    var_type_dic = {}
    for line in data_lines:
        valid_line = re.search('(.*);', line)
        if valid_line:
            valid_line = valid_line.group(1)
            sep_line = re.search('[ ]*([^ ]*)[ ]*([^ \[\]]*)', valid_line)
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
    data_patch = re.search('parameters[ ]*{([^{|^}]*)}', model_str)
    data_str = data_patch.group(1)
    data_lines = data_str.split('\n')
    var_type_dic = {}
    parameter_names = []
    for line in data_lines:
        valid_line = re.search('(.*);', line)
        if valid_line:
            valid_line = valid_line.group(1)
            type_str = re.search('[ ]*([^ <>\[\]]*).*', valid_line)
            range_str = re.search('[^<>]*<([^<>]*)>.*', valid_line)
            size_str = re.search('[^\[\]]*\[([^\[\]]*)\].*', valid_line)
            var_str = re.search('.* ([^ ]*)[ ]*.*', valid_line)
            if (type_str is not None) and (var_str is not None):
                type_str = type_str.group(1)
                var_str = var_str.group(1)
                var_type_dic[var_str] = type_str
                parameter_names.append(var_str)

    return parameter_names


def run_ZVCV(file_dir):
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.09/lifedead'
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.03/GLM_Binomial'
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.03/GLM_Poisson'
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.05/ssm'
    # file_dir = 'performance-tests-cmdstan/example-models/basic_estimators/bernoulli'
    robjects.globalenv.clear()
    # Assume the stan model file ends with .stan
    model_file = file_dir + '.stan'
    # Assume the data file ends with .data.R
    data_file = file_dir + '.data.R'
    # read data into env
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

    # run stan
    sm = pystan.StanModel(file=model_file)
    data, parameter_names = verifyDataType(sm, data)
    parameter_names = getParameterNames(sm)
    try:
        fit = sm.sampling(data=data, chains=1, iter=1000, verbose=True)

        # Extract parameters
        parameter_extract = fit.extract()
        parameter_values = []
        for parameter_name in parameter_names:
            parameter_values.append(parameter_extract[parameter_name])
        mcmc_samples = np.asarray(parameter_values)
        mcmc_samples = mcmc_samples.T

        # Unconstraint mcmc samples.
        unconstrain_mcmc_samples = []
        for i in range(mcmc_samples.shape[0]):
            tmp_dict = {}
            for j, parameter_name in enumerate(parameter_names):
                tmp_dict[parameter_name] = mcmc_samples[i, j]
            unconstrain_mcmc_samples.append(fit.unconstrain_pars(tmp_dict))
        unconstrain_mcmc_samples = np.asarray(unconstrain_mcmc_samples)

        # Calculate gradients of the log-probability
        # In this case, it seems unconstraint and constraint parameters are the same.
        num_of_iter = mcmc_samples.shape[0]
        grad_log_prob_val = []
        for i in range(num_of_iter):
            grad_log_prob_val.append(fit.grad_log_prob(unconstrain_mcmc_samples[i], adjust_transform=False))
        grad_log_prob_val = np.asarray(grad_log_prob_val)

        # Run control variates
        cv_linear_mcmc_samples = control_variate_linear(unconstrain_mcmc_samples, grad_log_prob_val)
        cv_quad_mcmc_samples = control_variate_quadratic(unconstrain_mcmc_samples, grad_log_prob_val)
        plot_comparison(unconstrain_mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples, fig_name='normal.png', fig_path='/home/ubuntu/', fig_size=(8, 8))

    except RuntimeError as re:
        print(re)

    return fit
