# encoding: utf-8

import logging as lg
import os
from datetime import datetime

def initialize_logger():

    # create folder for logging
    logs_path = './logs/' # define the path
    try:
        os.mkdir(logs_path)
    except OSError:
        print('Creation of directory %s failed' % logs_path)
    else:
        print("Successfully created log directory")

    # rename each log depending on the timeout

    current = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = current + '.log'
    currentLog_path = logs_path + log_name

    # log parameters

    lg.basicConfig(filename=currentLog_path, format = '%(asctime)s - %(levelname)s: %(message)s', level = lg.DEBUG)

    lg.info('Log initialized')
