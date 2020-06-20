import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Our objective is to identify as fast as possible the AD that has click through rate

# Simulation ads clicking data set
df = pd.read_csv('../Ads_CTR_Optimisation.csv')

print(df.head())

# UCB Algorithm
n_users = 10000
n_ads = 10
ads_selected = []
N_i = [0] * n_ads
R_i = [0] * n_ads
total_reward = 0

