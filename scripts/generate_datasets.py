# pylint: disable=line-too-long
"""
Script for generating synthetic datasets
"""

import os
import random
import math
import numpy as np

from src.settings import DATA_ROOT

dataset0 = np.empty([100, 53, 140])
# print(dataset1.dtype)

T = 140

mask0 = np.zeros(dataset0.shape)
label0 = np.zeros(dataset0.shape[0])
for j in range(100):
    for i in range(53):
        noise = np.random.normal(0, 1, T)
        C_i = random.uniform(-4, 4)
        a_i = random.uniform(1 / 14, 10 / 14)
        omega_i = 2 * math.pi * random.uniform(1, 70) / 140
        phi_i = random.uniform(0, 2 * math.pi)
        for t in range(T):
            # value = C_i + a_i * (t - T/2) + sin( omega_i * t + phi_i) + NOISE for t < T/2;
            # C_i + a_i * (T/2 - t) + sin( omega_i * t + phi_i) + NOISE for t > T/2;
            # C_i - random bias, a_i - random slope, omega_i - random frequency, phi_i - random initial phase
            if t < T / 2:
                dataset0[j][i][t] = (
                    C_i + a_i * (t - T / 2) + math.sin(omega_i * t + phi_i) + noise[t]
                )
            elif t >= T / 2:
                dataset0[j][i][t] = (
                    C_i + a_i * ((T / 2) - t) + math.sin(omega_i * t + phi_i) + noise[t]
                )
        mask0[j][i][int(T / 2)] = 1

dataset1 = np.empty([100, 53, 140])
mask1 = np.zeros(dataset1.shape)
label1 = np.ones(dataset1.shape[0])
for j in range(100):
    sign = np.sign(random.uniform(0, 1))
    random_shift = int(sign * random.uniform(10, 20))
    for i in range(53):
        noise = np.random.normal(0, 1, T)
        C_i = random.uniform(-4, 4)
        a_i = random.uniform(1 / 14, 10 / 14)
        omega_i = 2 * math.pi * random.uniform(1, 70) / 140
        phi_i = random.uniform(0, 2 * math.pi)
        for t in range(T):
            # value = C_i + a_i * (t - T/2) + sin( omega_i * t + phi_i) + NOISE for t < T/2;
            # C_i + a_i * (T/2 - t) + sin( omega_i * t + phi_i) + NOISE for t > T/2;
            # C_i - random bias, a_i - random slope, omega_i - random frequency, phi_i - random initial phase
            if t < T / 2 + random_shift:
                dataset1[j][i][t] = (
                    C_i
                    + a_i * (t - T / 2 - random_shift)
                    + math.sin(omega_i * t + phi_i)
                    + noise[t]
                )
            elif t >= T / 2 + random_shift:
                dataset1[j][i][t] = (
                    C_i
                    + a_i * (T / 2 + random_shift - t)
                    + math.sin(omega_i * t + phi_i)
                    + noise[t]
                )
        mask1[j][i][int(T / 2 + random_shift)] = 1

data = np.concatenate((dataset0, dataset1))
mask = np.concatenate((mask0, mask1))
label = np.concatenate((label0, label1))

os.makedirs(f"{DATA_ROOT}/synth1", exist_ok=True)
np.savez_compressed(f"{DATA_ROOT}/synth1/data.npz", data=data, labels=label, masks=mask)

###########################################################################################

dataset0 = np.empty([100, 53, 140])
T = 140

mask0 = np.zeros(dataset0.shape)
label0 = np.zeros(dataset0.shape[0])
for j in range(100):
    for i in range(53):
        noise = np.random.normal(0, 1, T)
        C_i = random.uniform(-4, 4)
        a_i = random.uniform(1 / 14, 10 / 14)
        omega_i = 2 * math.pi * random.uniform(1, 70) / 140
        phi_i = random.uniform(0, 2 * math.pi)
        for t in range(T):
            # value = C_i + a_i * (t - T/2) + sin( omega_i * t + phi_i) + NOISE for t < T/2;
            # C_i + a_i * (T/2 - t) + sin( omega_i * t + phi_i) + NOISE for t > T/2;
            # C_i - random bias, a_i - random slope, omega_i - random frequency, phi_i - random initial phase
            if t % 3 == 0:
                dataset0[j][i][t] = (
                    C_i + a_i * (t - T / 2) + math.sin(omega_i * t + phi_i) + noise[t]
                )
            elif (t % 3 == 1) or (t % 3 == 2):
                dataset0[j][i][t] = (
                    C_i + a_i * ((T / 2) - t) + math.sin(omega_i * t + phi_i) + noise[t]
                )
        mask0[j][i][int(T / 2)] = 1

dataset1 = np.empty([100, 53, 140])
mask1 = np.zeros(dataset1.shape)
label1 = np.ones(dataset1.shape[0])
for j in range(100):
    sign = np.sign(random.uniform(0, 1))
    random_shift = 0
    for i in range(53):
        noise = np.random.normal(0, 1, T)
        C_i = random.uniform(-4, 4)
        a_i = random.uniform(1 / 14, 10 / 14)
        omega_i = 2 * math.pi * random.uniform(1, 70) / 140
        phi_i = random.uniform(0, 2 * math.pi)
        for t in range(T):
            # value = C_i + a_i * (t - T/2) + sin( omega_i * t + phi_i) + NOISE for t < T/2;
            # C_i + a_i * (T/2 - t) + sin( omega_i * t + phi_i) + NOISE for t > T/2;
            # C_i - random bias, a_i - random slope, omega_i - random frequency, phi_i - random initial phase
            if t % 3 == 0:
                dataset1[j][i][t] = (
                    C_i
                    + a_i * (t - T / 2 - random_shift)
                    + math.sin(omega_i * t + phi_i)
                    + noise[t]
                )
            elif (t % 3 == 1) or (t % 3 == 2):
                dataset1[j][i][t] = (
                    C_i
                    + a_i * (T / 2 + random_shift - t)
                    + math.sin(omega_i * t + phi_i)
                    + noise[t]
                )

            if t == 2 * i:
                dataset1[j][i][t] = -dataset1[j][i][t]
        mask1[j][i][int(T / 2 + random_shift)] = 1

data = np.concatenate((dataset0, dataset1))
mask = np.concatenate((mask0, mask1))
label = np.concatenate((label0, label1))

os.makedirs(f"{DATA_ROOT}/synth2", exist_ok=True)
np.savez_compressed(f"{DATA_ROOT}/synth2/data.npz", data=data, labels=label, masks=mask)
