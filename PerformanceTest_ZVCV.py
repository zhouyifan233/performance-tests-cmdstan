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
    for data_keys in data:
        if data_keys in var_type_dic:
            claimed_type = var_type_dic[data_keys]
            if claimed_type.startswith('int'):
                data[data_keys] = np.int32(data[data_keys])
        else:
            print("verify data type failed! Didn't extract data type from stan-model successfully.")

    return data


def run_ZVCV(file_dir):
    robjects.globalenv.clear()
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.09/lifedead'
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.03/GLM_Binomial'
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.03/GLM_Poisson'
    # file_dir = 'performance-tests-cmdstan/example-models/BPA/Ch.05/ssm'
    # file_dir = 'performance-tests-cmdstan/example-models/basic_estimators/bernoulli'
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
    data = verifyDataType(sm, data)
    try:
        fit = sm.sampling(data=data, chains=1, iter=1000, verbose=True)
    except RuntimeError as re:
        print(re)

    return fit
