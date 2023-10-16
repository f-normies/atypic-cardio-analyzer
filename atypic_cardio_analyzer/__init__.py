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
from scipy.signal import savgol_filter
from math import sqrt
from multiprocessing import Pool

def open_txt(file, step=1, separator='\t', header=None):
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
    t = data['time'].values[::step]
    v = data['voltage'].values[::step]

    return np.asanyarray(t), np.asanyarray(v)

def preprocess(file, time_factor=1000, voltage_factor=1000, step=1, window=51, polyorder=3):
    t, v = open_txt(file, step=step)

    t = t * time_factor
    v = v * voltage_factor
    v = savgol_filter(v, window, polyorder)

    return t, v

def preprocess_opened(t, v, time_factor=1000, voltage_factor=1000, step=1, window=51, polyorder=3):
    t = t[::step] * time_factor
    v = v[::step] * voltage_factor
    v = savgol_filter(v, window, polyorder)

    return t, v

#Legacy, will be deleted at release
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

def adaptive_threshold(t, v, k=10):
    voltage_derivative = np.diff(v) / np.diff(t)
    normalized_derivative = (voltage_derivative - np.min(voltage_derivative)) / (np.max(voltage_derivative) - np.min(voltage_derivative))
    median_derivative = np.median(normalized_derivative)
    alpha = k * median_derivative
    return alpha

def estimate_refractory_period(t, v, alpha):
    action_potentials = find_action_potentials(t, v, alpha=alpha, refractory_period=0)
    
    intervals = np.diff([ap['start'] for ap in action_potentials])
    
    counts, bin_edges = np.histogram(intervals, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    threshold = np.max(counts) * 0.1
    rise_index = np.where(counts > threshold)[0][0]
    
    return bin_centers[rise_index]

def find_action_potentials(time, voltage, alpha=1.3, beta=0.0005, beta_tolerance=0.00025, refractory_period=60, offset=0):
    voltage_derivative = np.diff(voltage) / np.diff(time)

    candidate_phase_0_start_indices = np.where(voltage_derivative > alpha)[0]
    time_diff_candidates = np.diff(time[candidate_phase_0_start_indices])

    phase_0_start_indices = [candidate_phase_0_start_indices[0]]
    for i in range(1, len(candidate_phase_0_start_indices)):
        if time_diff_candidates[i-1] > refractory_period:
            phase_0_start_indices.append(candidate_phase_0_start_indices[i])
    phase_0_start_indices = np.array(phase_0_start_indices)

    action_potentials = []
    last_end_time = None
    for i, start_index in enumerate(phase_0_start_indices):
        start_time = time[start_index]
        if last_end_time is not None and (start_time - last_end_time) < refractory_period:
            continue

        pre_start_index = action_potentials[-1]['end'] if i > 0 else 0

        if i < len(phase_0_start_indices) - 1:
            next_start_index = phase_0_start_indices[i + 1]
            peak_index = np.argmax(voltage[start_index:next_start_index]) + start_index

            peak_to_next_start_voltage_derivative = np.diff(voltage[peak_index:next_start_index]) / np.diff(time[peak_index:next_start_index])
            end_index_candidates = np.where((beta - beta_tolerance < peak_to_next_start_voltage_derivative) & (peak_to_next_start_voltage_derivative < beta + beta_tolerance))[0]

            if end_index_candidates.size > 0:
                end_index = end_index_candidates[0] + peak_index
            else:
                end_index = np.argmin(voltage[peak_index:next_start_index]) + peak_index
        else:
            peak_index = np.argmax(voltage[start_index:]) + start_index

            peak_to_end_voltage_derivative = np.diff(voltage[peak_index:]) / np.diff(time[peak_index:])
            end_index_candidates = np.where((beta - beta_tolerance < peak_to_end_voltage_derivative) & (peak_to_end_voltage_derivative < beta + beta_tolerance))[0]

            if end_index_candidates.size > 0:
                end_index = end_index_candidates[0] + peak_index
            else:
                end_index = np.argmin(voltage[peak_index:]) + peak_index

        action_potentials.append({
            'pre_start': pre_start_index,
            'start': start_index - offset,
            'peak': peak_index,
            'end': end_index
        })

        last_end_time = time[end_index]

    return action_potentials

def find_voltage_speed(ap, time, voltage):
    prestart_index = ap['pre_start']
    start_index = ap['start']
    peak_index = ap['peak']
    end_index = ap['end']

    phase_4_time = time[prestart_index:start_index]
    phase_4_voltage = voltage[prestart_index:start_index]
    phase_4_speed = np.diff(phase_4_voltage)/np.diff(phase_4_time)

    phase_0_time = time[start_index:peak_index]
    phase_0_voltage = voltage[start_index:peak_index]
    phase_0_speed = np.diff(phase_0_voltage)/np.diff(phase_0_time)

    return np.average(phase_4_speed), np.max(phase_0_speed)

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
    x = np.array(time)
    y = np.array(voltage)
    n = 1
    while rad > avr_rad:
        x = x[:int(len(y)/2**n)]
        y = y[:int(len(y)/2**n)]
        n+=1
        ma1 = nearest_value(x, y, x[np.argmax(y)], np.min(y))
        if ma1 + 1 < len(x):
            rad, x_r, y_r = radius(x[ma1], y[ma1], x[ma1 + 1], y[ma1 + 1], x[ma1 - 1], y[ma1 - 1])
        elif ma1 - 1 > 0:
            ma1 -= 1
            rad, x_r, y_r = radius(x[ma1], y[ma1], x[ma1 + 1], y[ma1 + 1], x[ma1 - 1], y[ma1 - 1])
        if n > 4:
            break
    return rad, x_r, y_r

def save_file(args):
    number, current_time, current_voltage, destination = args
    filename = os.path.join(destination, f"{number + 1}.txt")
    
    with open(filename, 'w') as f:
        for t, v in zip(current_time, current_voltage):
            f.write(f"{t}\t{v}\n")

def save_aps_to_txt(destination, aps, time, voltage):
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)

    time_intervals = []

    args = []
    for number, ap in enumerate(aps):
        current_time = time[ap['pre_start']:ap['end']]
        current_voltage = voltage[ap['pre_start']:ap['end']]
        time_intervals.append([current_time[0], current_time[-1]])
        
        args.append((number, current_time, current_voltage, destination))

    with Pool() as pool:
        pool.map(save_file, args)

    return time_intervals

def save_aps_to_xlsx(destination, aps, time, voltage, original_time=None, original_voltage=None, limit_rad=1000):
    data = []

    for number, ap in enumerate(aps):
        if original_time and original_voltage:
            phase_4_speed, phase_0_speed = find_voltage_speed(ap, original_time, original_voltage)
        else:
            phase_4_speed, phase_0_speed = find_voltage_speed(ap, time, voltage)

        current_ap_time = time[ap['pre_start']:ap['end']]
        current_ap_voltage = voltage[ap['pre_start']:ap['end']]

        radius, x, y = circle(current_ap_time, current_ap_voltage, limit_rad)

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