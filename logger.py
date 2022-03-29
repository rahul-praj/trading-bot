#!/usr/birn/#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging as lg

lg.basicConfig(filename='trading_bot.log', format = '%(asctime)s: %(message)s', level = lg.DEBUG)

# logging levels: DEBUG, INFO, WARNING, ERROR

lg.info('This is an info message')
lg.debug('This is a debugging message')
lg.warning('This is a warning!')
lg.error('This is an error message')
