from PerformanceTest_ZVCV import run_ZVCV
import pandas as pd
import numpy as np
import sys
import os
import subprocess
from os.path import exists


# Parameters
# output_path = 'performance-tests-cmdstan/ZVCV_comparison/'
if 1:
    example_list_path = 'list_of_examples_2.txt'
    output_path = 'ZVCV-results-2/'
    terminal_path = 'ZVCV-messages-2/terminal-messages/'
    error_path = 'ZVCV-messages-2/error-messages/'
else:
    example_list_path = 'list_of_examples_1.txt'
    output_path = 'ZVCV-results-1/'
    terminal_path = 'ZVCV-messages-1/terminal-messages/'
    error_path = 'ZVCV-messages-1/error-messages/'

# Read list of examples
list_examples = []
with open(example_list_path, 'r') as f:
    example_path = f.readline()
    while example_path:
        example_path = example_path.strip()
        list_examples.append(example_path)
        example_path = f.readline()
# print(list_examples)

# run test
start_i = 0    # 390 cv issue; 392; 434
for i, example_path in enumerate(list_examples):
    print('Index: ' + str(i))
    print('Example Path: ' + example_path)

    file_dir = example_path.replace('performance-tests-cmdstan/', '')
    output_filename = file_dir.replace('/', '@')

    if exists(output_path + output_filename + '.csv') is False:
        terminal_m_output = open(terminal_path + output_filename + '.txt', 'w')
        error_m_output = open(error_path + output_filename + '.txt', 'w')
        subprocess.run(['python', 'PerformanceTest_ZVCV.py', file_dir, output_path], stdout=terminal_m_output, stderr=error_m_output, text=True)
        terminal_m_output.close()
        error_m_output.close()


