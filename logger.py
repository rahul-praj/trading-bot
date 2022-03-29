#!/usr/birn/#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging as lg
import os
from datetime import datetime

# create folder for logging
path = './logs'
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

# logging levels: DEBUG, INFO, WARNING, ERROR

lg.info('This is an info message')
lg.debug('This is a debugging message')
lg.warning('This is a warning!')
lg.error('This is an error message')
