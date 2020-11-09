#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

def convertDateStrToYearJulian(date):
    
    fmt = '%Y-%m-%d'
    dt = datetime.datetime.strptime(date, fmt)

    #convert to time tuple for converting to julian day
    tt = dt.timetuple()
    jday = tt.tm_yday
    year = tt.tm_year

    return jday, year


