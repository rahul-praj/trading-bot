#!/usr/birn/#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging as lg
import os
from datetime import datetime

def initialize_logger():


    # create folder for logging
    path = './logs' # define the path
    try:
        os.mkdir(path)
    except:
        print('Creation of directory %s failed' % path)

    # rename each log depending on the timeout

    current = datetime.now().stfrtime("%Y%m%d_%H:%M:%S")
    log_name = current + '.log'
    currentLog_path = logs_path + log_name

    # log parameters

    lg.basicConfig(filename=log_name, format = '%(asctime)s: %(message)s', level = lg.DEBUG)
    lg.getLogger().addHandler(lg.StreamHandler())

    lg.info('Log initialized')
