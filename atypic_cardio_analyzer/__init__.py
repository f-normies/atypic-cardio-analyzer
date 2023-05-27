# Copyright (C) 2023 Oleg Zakharov, Aleksei Aredov, Ilya Polusmak
#
# This file is part of atypic-cardio-analyzer library.
#
# atypic-cardio-analyzer library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import pandas as pd
import dask.dataframe as dd
import openpyxl
import os
import shutil
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from math import sqrt

def open_txt(file, separator='\t', header=None):
    column_names = ['time', 'voltage']
    column_types = {'time': np.float64, 'voltage': np.float64}
    
    try:
        data = dd.read_csv(file, engine='c', decimal=',', sep=separator, encoding='latin-1', 
                       on_bad_lines='skip', header=header, names=column_names,
                       dtype=column_types)
    except ValueError:
        data = dd.read_csv(file, engine='c', decimal='.', sep=separator, encoding='latin-1', 
                       on_bad_lines='skip', header=header, names=column_names,
                       dtype=column_types)

    data = data.compute()
    time = data['time'].values
    voltage = data['voltage'].values

    return time, voltage

def preprocess(file, step=1, sigma=5):
    time, voltage = open_txt(file)

    time = time[::step] * 1000
    voltage = voltage[::step] * 1000
    voltage = gaussian_filter(voltage, sigma=sigma)

    return time, voltage

def replace_nan_with_nearest(value_list, index):
    left, right = index - 1, index + 1

    try:
        while left >= 0 or right < len(value_list):
            if left >= 0 and not math.isnan(value_list[left]):
                return value_list[left]
            elif right < len(value_list) and not math.isnan(value_list[right]):
                return value_list[right]
            left -= 1
            right += 1
    finally:
        return 0

def find_action_potentials(time, voltage, alpha=0.9, refractory_period=280, offset=0):
    voltage_derivative = np.diff(voltage) / np.diff(time)

    candidate_phase_0_start_indices = np.where(voltage_derivative > alpha)[0]
    diff_candidates = np.diff(candidate_phase_0_start_indices)

    phase_0_start_indices = np.insert(candidate_phase_0_start_indices[1:][diff_candidates > refractory_period], 0, candidate_phase_0_start_indices[0])

    action_potentials = []
    for i, start_index in enumerate(phase_0_start_indices):
        pre_start_index = action_potentials[-1]['end'] if i > 0 else 0

        if i < len(phase_0_start_indices) - 1:
            next_start_index = phase_0_start_indices[i + 1]
            peak_index = np.argmax(voltage[start_index:next_start_index]) + start_index
            end_index = np.argmin(voltage[peak_index:next_start_index]) + peak_index
        else:
            peak_index = np.argmax(voltage[start_index:]) + start_index
            end_index = np.argmin(voltage[peak_index:]) + peak_index

        action_potentials.append({
            'pre_start': pre_start_index,
            'start': start_index - offset,
            'peak': peak_index,
            'end': end_index
        })

    return action_potentials

def max_slope(time, voltage):
    if len(time) != len(voltage):
        raise ValueError("time and voltage should be of the same length")

    if len(time) < 5:
        raise ValueError("time and voltage should be at least of length 5")

    max_slope = 0
    for i in range(2, len(time) - 2):
        # five-point linear regression
        A = np.vstack([time[i-2:i+3], np.ones(5)]).T
        m, c = np.linalg.lstsq(A, voltage[i-2:i+3], rcond=None)[0]
        max_slope = max(max_slope, m)
    return max_slope

def find_voltage_speed(ap, time, voltage):
    prestart_index = ap['pre_start']
    start_index = ap['start']
    peak_index = ap['peak']

    phase_4_time = time[prestart_index:start_index]
    phase_4_voltage = voltage[prestart_index:start_index]
    phase_4_speed = max_slope(phase_4_time, phase_4_voltage)

    phase_0_time = time[start_index:peak_index]
    phase_0_voltage = voltage[start_index:peak_index]
    phase_0_speed = max_slope(phase_0_time, phase_0_voltage)

    return 1000 * phase_4_speed, phase_0_speed

def circle(time, voltage, avr_rad=1000):
    def nearest_value(items_x, items_y, value_x, value_y):
        dist = np.sqrt((items_x - value_x) ** 2 + (items_y - value_y) ** 2)
        return np.argmin(dist)

    def flat(x, y, n):
        return x[::n], y[::n]

    def radius(x_c, y_c, x_1, y_1, x_2, y_2):
        cent1_x, cent1_y = (x_c + x_1) / 2, (y_c + y_1) / 2
        cent2_x, cent2_y = (x_c + x_2) / 2, (y_c + y_2) / 2

        k1 = (cent1_x - x_c) / (cent1_y - y_c)
        b1 = cent1_y + k1 * cent1_x

        k2 = (cent2_x - x_c) / (cent2_y - y_c)
        b2 = cent2_y + k2 * cent2_x

        x_r = (b2 - b1) / (k2 - k1)
        y_r = -k1 * x_r + b1

        rad = np.sqrt((x_r - x_c) ** 2 + (y_r - y_c) ** 2)
        return rad, x_r, y_r

    x = np.array(time)
    y = np.array(voltage)

    x = x[:np.argmax(y)]
    y = y[:np.argmax(y)]

    l = 8
    dff = flat(x, y, l)
    ma = nearest_value(dff[0], dff[1], x[np.argmax(y)], np.min(y))

    o = 10

    while ma + o >= len(dff[0]):
        l = l // 2
        if l == 0:
            return 10, -10, 0

        dff = flat(x, y, l)
        ma = nearest_value(dff[0], dff[1], x[np.argmax(y)], np.min(y))

    while dff[1][ma + o] - dff[1][ma] >= 8:
        if (dff[1][ma + o] - dff[1][ma]) / (dff[1][ma + (o - 1)] - dff[1][ma]) > 3.5:
            o -= 1
            ma = nearest_value(dff[0], dff[1], x[np.argmax(y)], np.min(y))
            break
        o -= 1
        ma = nearest_value(dff[0], dff[1], x[np.argmax(y)], np.min(y))

    rad, x_r, y_r = radius(dff[0][ma], dff[1][ma], dff[0][ma + o], dff[1][ma + o], dff[0][ma - o], dff[1][ma - o])
    k = 0
    try:
        while rad > avr_rad:
            k += 1
            ma -= 5
            rad_new, x_r_new, y_r_new = radius(dff[0][ma], dff[1][ma], dff[0][ma + o], dff[1][ma + o], dff[0][ma - o],
                                               dff[1][ma - o])
            if k > 100:
                break
        if rad_new < rad:
            return rad_new, x_r_new, y_r_new
        else:
            return rad, x_r, y_r

    except Exception as e:
        return rad, x_r, y_r

def save_aps_to_txt(destination, aps, time, voltage):
    if os.path.exists(destination):
        shutil.rmtree(destination)

    os.makedirs(destination)

    time_intervals = []

    for number, ap in enumerate(aps):
        current_time = time[ap['pre_start']:ap['end']]
        current_voltage = voltage[ap['pre_start']:ap['end']]

        time_intervals.append([current_time[0], current_time[-1]])

        data = np.column_stack((current_time, current_voltage))

        np.savetxt(os.path.join(destination, f"{number + 1}.txt"), data, delimiter='\t')

    return time_intervals

def save_aps_to_xlsx(destination, aps, time, voltage, limit_rad=1000):
    data = []
    radius_list_local = []
    phase_4_speed_list_local = []
    phase_0_speed_list_local = []

    for number, ap in enumerate(aps):
        phase_4_speed, phase_0_speed = find_voltage_speed(ap, time, voltage)

        if math.isnan(phase_4_speed):
            phase_4_speed = replace_nan_with_nearest(phase_4_speed_list_local, number)
        if math.isnan(phase_0_speed):
            phase_0_speed = replace_nan_with_nearest(phase_0_speed_list_local, number)

        current_ap_time = time[ap['pre_start']:ap['end']]
        current_ap_voltage = voltage[ap['pre_start']:ap['end']]

        radius, x, y = circle(current_ap_time, current_ap_voltage, limit_rad)

        if math.isnan(radius):
            radius = replace_nan_with_nearest(radius_list_local, number)

        radius_list_local.append(radius)


        row = {
            "Номер ПД": number + 1,
            "Начало": f"{current_ap_time[0]:.2f}",
            "Конец": f"{current_ap_time[-1]:.2f}",
            "Радиус": round(radius, 2),
            "dV/dt4": round(phase_4_speed, 3),
            "dV/dt0": round(phase_0_speed, 3)
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_excel(destination, index=False, engine='openpyxl')