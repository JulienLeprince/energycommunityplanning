import pulp
import pandas as pd
import numpy as np
import time
import os
from datetime import date
import sys

arg_nb_of_buildings = 'all'  # 'all' or int
arg_subset_of_scenarios = 'all'  # 'all' or int
arg_folder_name = 'demo_folder'

# Loading parameters
from src.parameters import *
# Loading RC models
from src.RC_models import *

# Path definition
path_gitrepo = r'C:/Users/20190285/Documents/GitHub/energycommunityplanning/'
path_in = path_gitrepo + '/data/in/'
path_out = path_gitrepo + '/data/out/'
version = 'demo'

# RC building models
file_RCmodels = 'https://github.com/JulienLeprince/greybrickbuildings/blob/main/data/calibrated_models.csv?raw=true'
df_RC = pd.read_csv(file_RCmodels, index_col='identification_number')
df_RC = df_RC[df_RC['nCPBES'] < 0.003]  # filtering RC models based on nCPBES values - here covering good and close model fits
buildings = list(df_RC.index)

# Stochastic scenario definition
probabilities = pd.read_csv(path_in+'scenario_probabilities.csv', index_col=[0])
scenarios = probabilities.shape[0]

# Number of considered scenarios redefined here
if arg_subset_of_scenarios != 'all':
    scenarios = int(arg_subset_of_scenarios)

# Reading input data
dfw, dfb = dict(), dict()
p_elec, p_gas = dict(), dict()
for s in range(scenarios):
    dfw[s] = pd.read_csv(path_in+'scenario_'+ str(s) +'.csv', usecols=[1,2])
    dfw[s].rename(columns = {'Ta':'T_a', 'Ps':'Q_sol'}, inplace = True)
    dfw[s] = dfw[s].round(decimals=4)

    dfb[s] = dict()
    bi = 0
    for b in range(len(df_RC.index)):
        data = pd.read_csv(path_in+'/scenario_'+ str(s) +'.csv', usecols=[5+bi, 6+bi])
        uuid = int(data.columns[0].split('_')[0])
        dfb[s][uuid] = data
        dfb[s][uuid].rename(columns={str(uuid)+'_T_blg_set':'T_blg_set', str(uuid)+'_E_elec':'E_blg'}, inplace = True)
        dfb[s][uuid]['E_blg'] = dfb[s][uuid]['E_blg']/1000  # converting from W to kW
        dfb[s][uuid] = dfb[s][uuid].round(decimals=4)
        bi += 2

    p_elec[s] = pd.read_csv(path_in+'/scenario_'+ str(s) +'.csv', usecols=[3])
    p_elec[s] = p_elec[s]['Day-ahead Price [EUR/kWh]'] + [tax_electricity+tariff_elec_transmission_and_distribution]*len(p_elec[s]['Day-ahead Price [EUR/kWh]'])
    p_gas[s] = pd.read_csv(path_in+'/scenario_'+ str(s) +'.csv', usecols=[4])
    p_gas[s] = p_gas[s]['gas_price [EUR/kWh]'] + [tax_naturalgas]*len(p_gas[s]['gas_price [EUR/kWh]'])
    p_elec[s] = p_elec[s].round(decimals=4)
    p_gas[s] = p_gas[s].round(decimals=4)

H = p_gas[s].shape[0]
buildings = list(dfb[s].keys())
buildings = [value for value in buildings if value in df_RC.index]

# Number of considered buildings redefined here
if arg_nb_of_buildings != 'all':
    buildings = buildings[0:int(arg_nb_of_buildings)]

# Calculating heat pump COP
for s in range(scenarios):
    for b in buildings:
        dfb[s][b]['COP_hp'] = pi_hp_1 * np.exp(pi_hp_2 * (dfb[s][b]['T_blg_set'] - dfw[s]['T_a'])) \
                           + pi_hp_3 * np.exp(pi_hp_4 * (dfb[s][b]['T_blg_set'] - dfw[s]['T_a']))


# Defining LP problem & variables
my_lp_problem = pulp.LpProblem('My_LP_Problem', pulp.LpMinimize)

# Variables - building block
# Continous
T_blg = pulp.LpVariable.dicts('var_T_blg', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continuous')  # Building inside temperature
O_slk_blg = pulp.LpVariable.dicts('var_O_slk_blg', (range(scenarios), buildings, range(H)), lowBound=0, cat='Continuous')
E_blg_bat = pulp.LpVariable.dicts('var_E_blg_bat', (range(scenarios), buildings, range(H + 1)), cat='Continuous')
Q_tes = pulp.LpVariable.dicts('var_Q_tes', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continuous')
Q_stc = pulp.LpVariable.dicts('var_Q_stc', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continuous')
Q_sp = pulp.LpVariable.dicts('var_Q_sp', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
Q_tes_ch = pulp.LpVariable.dicts('var_Q_tes_ch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
Q_tes_dch = pulp.LpVariable.dicts('var_Q_tes_dch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
Q_hp = pulp.LpVariable.dicts('var_Q_hp', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
Q_bol = pulp.LpVariable.dicts('var_Q_bol', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
E_blg_bat_ch = pulp.LpVariable.dicts('var_E_blg_bat_ch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
E_blg_bat_dch = pulp.LpVariable.dicts('var_E_blg_bat_dch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
E_blg_hp = pulp.LpVariable.dicts('var_E_blg_hp', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
E_blg_pv = pulp.LpVariable.dicts('var_E_blg_pv', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
E_blg_in = pulp.LpVariable.dicts('var_E_blg_in', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
E_blg_out = pulp.LpVariable.dicts('var_E_blg_out', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
slk_blg_in = pulp.LpVariable.dicts('var_slk_blg_in', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
slk_blg_out = pulp.LpVariable.dicts('var_slk_blg_out', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
V_blg_gas = pulp.LpVariable.dicts('var_V_blg_gas', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Continous')
# Sizing
C_blg_hp = pulp.LpVariable.dicts('var_C_blg_hp', (range(scenarios), buildings), lowBound=0, upBound=C_blg_hp_max, cat='Continous')
C_blg_bat = pulp.LpVariable.dicts('var_C_blg_bat', (range(scenarios), buildings), lowBound=0, upBound=C_blg_bat_max, cat='Continous')
C_blg_tes = pulp.LpVariable.dicts('var_C_blg_tes', (range(scenarios), buildings), lowBound=0, upBound=C_blg_tes_max, cat='Continous')
A_blg_stc = pulp.LpVariable.dicts('var_A_blg_stc', (range(scenarios), buildings), lowBound=0, upBound=A_blg_stc_max, cat='Continous')
A_blg_pv = pulp.LpVariable.dicts('var_A_blg_pv', (range(scenarios), buildings), lowBound=0, upBound=A_blg_pv_max, cat='Continous')
C_blg_bol = pulp.LpVariable.dicts('var_C_blg_bol', (range(scenarios), buildings), lowBound=0, upBound=C_blg_bol_max, cat='Continous')
i_blg_hp = pulp.LpVariable.dicts('var_i_blg_hp', (range(scenarios), buildings), lowBound=0, cat='Binary')
i_blg_bat = pulp.LpVariable.dicts('var_i_blg_bat', (range(scenarios), buildings), lowBound=0, cat='Binary')
i_blg_tes = pulp.LpVariable.dicts('var_i_blg_tes', (range(scenarios), buildings), lowBound=0, cat='Binary')
i_blg_bol = pulp.LpVariable.dicts('var_i_blg_bol', (range(scenarios), buildings), lowBound=0, cat='Binary')
i_blg_stc = pulp.LpVariable.dicts('var_i_blg_stc', (range(scenarios), buildings), lowBound=0, cat='Binary')
i_blg_pv = pulp.LpVariable.dicts('var_i_blg_pv', (range(scenarios), buildings), lowBound=0, cat='Binary')
# # Other
# i_blg_bat_ch = pulp.LpVariable.dicts('var_i_blg_bat_ch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Binary')
# i_blg_bat_dch = pulp.LpVariable.dicts('var_i_blg_bat_dch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Binary')
# i_blg_tes_ch = pulp.LpVariable.dicts('var_i_blg_tes_ch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Binary')
# i_blg_tes_dch = pulp.LpVariable.dicts('var_i_blg_tes_dch', (range(scenarios), buildings, range(H + 1)), lowBound=0, cat='Binary')

# Variables community
# Continous
E_hv_in = pulp.LpVariable.dicts('var_E_net', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
E_mv_out = pulp.LpVariable.dicts('var_E_mv_out', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
slk_mv_out = pulp.LpVariable.dicts('var_slk_mv_out', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
E_mv_in = pulp.LpVariable.dicts('var_E_mv_in', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
slk_mv_in = pulp.LpVariable.dicts('var_slk_mv_in', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
E_com_pv = pulp.LpVariable.dicts('var_E_com_pv', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
E_com_bat = pulp.LpVariable.dicts('var_E_com_bat', (range(scenarios), range(H + 1)), lowBound=0, cat='Continuous')
E_com_bat_ch = pulp.LpVariable.dicts('var_E_com_bat_ch', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
E_com_bat_dch = pulp.LpVariable.dicts('var_E_com_bat_dch', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
E_com_hyd = pulp.LpVariable.dicts('var_E_com_hyd', (range(scenarios), range(H + 1)), lowBound=0, cat='Continuous')
E_com_hyd_ch = pulp.LpVariable.dicts('var_E_com_hyd_ch', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
E_com_hyd_dch = pulp.LpVariable.dicts('var_E_com_hyd_dch', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
# Sizing
A_com_pv = pulp.LpVariable.dicts('var_C_com_pv', range(scenarios), lowBound=0, upBound=A_com_pv_max, cat='Continous')
C_com_bat = pulp.LpVariable.dicts('var_C_com_bat', range(scenarios), lowBound=0, upBound=C_com_bat_max, cat='Continous')
C_com_hyd = pulp.LpVariable.dicts('var_C_com_hyd', range(scenarios), lowBound=0, upBound=C_com_hyd_max, cat='Continous')
C_com_elec = pulp.LpVariable.dicts('var_C_com_elec', range(scenarios), lowBound=0, upBound=C_com_elec_max, cat='Continous')
C_com_fc = pulp.LpVariable.dicts('var_C_com_fc', range(scenarios), lowBound=0, upBound=C_com_fc_max, cat='Continous')
i_com_bat = pulp.LpVariable.dicts('var_i_com_bat', range(scenarios), lowBound=0, cat='Binary')
i_com_hyd = pulp.LpVariable.dicts('var_i_com_hyd', range(scenarios), lowBound=0, cat='Binary')
i_com_pv = pulp.LpVariable.dicts('var_i_com_pv', range(scenarios), lowBound=0, cat='Binary')
# # Other
# i_com_bat_ch = pulp.LpVariable.dicts('var_i_com_bat_ch', (range(scenarios), range(H)), lowBound=0, cat='Binary')
# i_com_bat_dch = pulp.LpVariable.dicts('var_i_com_bat_dch', (range(scenarios), range(H)), lowBound=0, cat='Binary')
# i_com_hyd_ch = pulp.LpVariable.dicts('var_i_com_hyd_ch', (range(scenarios), range(H)), lowBound=0, cat='Binary')
# i_com_hyd_dch = pulp.LpVariable.dicts('var_i_com_hyd_dch', (range(scenarios), range(H)), lowBound=0, cat='Binary')

# Variables Objective function
O_tot = pulp.LpVariable.dicts('var_O_tot', range(scenarios), lowBound=0, cat='Continous')
O_opr = pulp.LpVariable.dicts('var_O_opr', range(scenarios), lowBound=0, cat='Continous')
O_co2 = pulp.LpVariable.dicts('var_O_co2', range(scenarios), lowBound=0, cat='Continous')
O_inv = pulp.LpVariable.dicts('var_O_inv', range(scenarios), lowBound=0, cat='Continous')
O_slk = pulp.LpVariable.dicts('var_O_slk', range(scenarios), lowBound=0, cat='Continous')
p_blg_bat = pulp.LpVariable.dicts('var_p_blg_bat', (range(scenarios), buildings), lowBound=0, cat='Continous')
p_blg_tes = pulp.LpVariable.dicts('var_p_blg_tes', (range(scenarios), buildings), lowBound=0, cat='Continous')
p_blg_hp = pulp.LpVariable.dicts('var_p_blg_hp', (range(scenarios), buildings), lowBound=0, cat='Continous')
p_blg_bol = pulp.LpVariable.dicts('var_p_blg_bol', (range(scenarios), buildings), lowBound=0, cat='Continous')
p_blg_pv = pulp.LpVariable.dicts('var_p_blg_pv', (range(scenarios), buildings), lowBound=0, cat='Continous')
p_blg_stc = pulp.LpVariable.dicts('var_p_blg_stc', (range(scenarios), buildings), lowBound=0, cat='Continous')
p_com_bat = pulp.LpVariable.dicts('var_p_com_bat', range(scenarios), lowBound=0, cat='Continous')
p_com_hyd = pulp.LpVariable.dicts('var_p_com_hyd', range(scenarios), lowBound=0, cat='Continous')
p_com_pv = pulp.LpVariable.dicts('var_p_com_pv', range(scenarios), lowBound=0, cat='Continous')

# System constraints
for s in range(scenarios):

    # Building block
    for b in buildings:
        # Building system
        my_lp_problem = RCmodel(my_lp_problem, df_RC.loc[b, 'model_name'], dfw[s], T_blg[s][b], Q_sp[s][b],
                                O_slk_blg[s][b], H, b, s, T_set=dfb[s][b]['T_blg_set'].iloc[0])
        # my_lp_problem = rc.RCmodel(my_lp_problem, df_RC.loc[b, 'model_name'], dfw[s], T_blg[s], Q_sp[s], H, b, s)

        for t in range(H):
            # my_lp_problem += T_blg[s][b][t+1] <= dfb[s][b]['T_blg_set'].iloc[t+1] + T_blg_buffer  # cooling boundary
            my_lp_problem += T_blg[s][b][t] >= dfb[s][b]['T_blg_set'].iloc[t] - T_blg_buffer  # heating boundary
            # Battery
            my_lp_problem += E_blg_bat[s][b][t + 1] == E_blg_bat[s][b][t] * decay_blg_bat \
                             + E_blg_bat_ch[s][b][t] * eff_blg_bat_ch \
                             - E_blg_bat_dch[s][b][t] * (1 / eff_blg_bat_dch)
            my_lp_problem += E_blg_bat[s][b][t] <= C_blg_bat[s][b]
            my_lp_problem += E_blg_bat[s][b][t] >= C_blg_bat_min * i_blg_bat[s][b]
            my_lp_problem += E_blg_bat_ch[s][b][t] <= C_blg_bat[s][b] * power_eff_blg_bat_ch
            my_lp_problem += E_blg_bat_dch[s][b][t] <= C_blg_bat[s][b] * power_eff_blg_bat_dch
            # my_lp_problem += i_blg_bat_ch[s][b][t] + i_blg_bat_dch[s][b][t] <= 1
            # my_lp_problem += E_blg_bat_ch[s][b][t] <= i_blg_bat_ch[s][b][t] * C_blg_bat_max
            # my_lp_problem += E_blg_bat_dch[s][b][t] <= i_blg_bat_dch[s][b][t] * C_blg_bat_max
            # Thermal energy storage
            my_lp_problem += Q_tes[s][b][t + 1] == Q_tes[s][b][t] * decay_blg_tes \
                             + Q_tes_ch[s][b][t] * eff_blg_tes_ch \
                             - Q_tes_dch[s][b][t] * (1 / eff_blg_tes_dch)
            my_lp_problem += Q_tes[s][b][t] <= C_blg_tes[s][b]
            my_lp_problem += Q_tes_ch[s][b][t] <= C_blg_tes[s][b] * power_eff_blg_tes_ch
            my_lp_problem += Q_tes_dch[s][b][t] <= C_blg_tes[s][b] * power_eff_blg_tes_dch
            # my_lp_problem += i_blg_tes_ch[s][b][t] + i_blg_tes_dch[s][b][t] <= 1
            # my_lp_problem += Q_tes_ch[s][b][t] <= i_blg_tes_ch[s][b][t] * C_blg_tes_max
            # my_lp_problem += Q_tes_dch[s][b][t] <= i_blg_tes_dch[s][b][t] * C_blg_tes_max
            # Heat pump
            my_lp_problem += Q_hp[s][b][t] == E_blg_hp[s][b][t] * dfb[s][b]['COP_hp'].iloc[t]
            my_lp_problem += Q_hp[s][b][t] <= C_blg_hp[s][b]
            # Boiler
            my_lp_problem += Q_bol[s][b][t] == V_blg_gas[s][b][t] * eff_blg_bol
            my_lp_problem += Q_bol[s][b][t] <= C_blg_bol[s][b]
            # Photovoltaics
            my_lp_problem += E_blg_pv[s][b][t] == A_blg_pv[s][b] * dfw[s]['Q_sol'].iloc[t] * eff_blg_pv
            # Solar thermal collector
            my_lp_problem += Q_stc[s][b][t] == A_blg_stc[s][b] * eff_blg_stc * (dfw[s]['Q_sol'].iloc[t]
                                                                                - U_blg_stc * (T_stc - dfw[s]['T_a'].iloc[t]))
            # Energy balance
            my_lp_problem += Q_sp[s][b][t] + Q_tes_ch[s][b][t] == Q_tes_dch[s][b][t] + Q_hp[s][b][t] + Q_bol[s][b][t]
            my_lp_problem += dfb[s][b]['E_blg'].iloc[t] + E_blg_bat_ch[s][b][t] + E_blg_hp[s][b][t] + E_blg_out[s][b][t] \
                             == E_blg_bat_dch[s][b][t] + E_blg_pv[s][b][t] + E_blg_in[s][b][t]
            # Building slk
            my_lp_problem += E_blg_in[s][b][t] <= E_lv_max + slk_blg_in[s][b][t]
            my_lp_problem += E_blg_out[s][b][t] <= E_lv_max + slk_blg_out[s][b][t]

        # Sizing
        my_lp_problem += C_blg_bat[s][b] <= C_blg_bat_max * i_blg_bat[s][b]
        my_lp_problem += C_blg_bat[s][b] >= C_blg_bat_min * i_blg_bat[s][b]
        my_lp_problem += C_blg_tes[s][b] <= C_blg_tes_max * i_blg_tes[s][b]
        my_lp_problem += C_blg_hp[s][b] <= C_blg_hp_max * i_blg_hp[s][b]
        my_lp_problem += C_blg_hp[s][b] >= C_blg_hp_min * i_blg_hp[s][b]
        my_lp_problem += C_blg_bol[s][b] <= C_blg_bol_max * i_blg_bol[s][b]
        my_lp_problem += C_blg_bol[s][b] >= C_blg_bol_min * i_blg_bol[s][b]
        my_lp_problem += A_blg_pv[s][b] <= A_blg_pv_max * i_blg_pv[s][b]
        my_lp_problem += A_blg_pv[s][b] >= A_blg_pv_min * i_blg_pv[s][b]
        my_lp_problem += A_blg_stc[s][b] <= A_blg_stc_max * i_blg_stc[s][b]
        my_lp_problem += A_blg_stc[s][b] >= A_blg_stc_min * i_blg_stc[s][b]
        my_lp_problem += A_blg_pv[s][b] + A_blg_stc[s][b] <= A_blg_roof_max
        # Initial conditions
        my_lp_problem += T_blg[s][b][0] == dfb[s][b]['T_blg_set'].iloc[0]
        my_lp_problem += E_blg_bat[s][b][0] <= E_blg_bat[s][b][H]
        my_lp_problem += Q_tes[s][b][0] <= Q_tes[s][b][H]
    for t in range(H):
        # Grid topology - energy balance
        my_lp_problem += sum(E_blg_out[s][bi][t] for bi in buildings) + E_mv_out[s][t] \
                         == sum(E_blg_in[s][bi][t] for bi in buildings) + E_mv_in[s][t]
        # Energy community - energy balance
        my_lp_problem += E_mv_out[s][t] + E_com_bat_ch[s][t] + E_com_hyd_ch[s][t] \
                         == E_com_bat_dch[s][t] + E_com_hyd_dch[s][t] + E_com_pv[s][t] + E_mv_in[s][t] + E_hv_in[s][t]
        # Slack community
        my_lp_problem += E_mv_out[s][t] <= E_mv_max + slk_mv_out[s][t]
        my_lp_problem += E_mv_in[s][t] <= E_mv_max + slk_mv_in[s][t]

        # Battery
        my_lp_problem += E_com_bat[s][t + 1] == E_com_bat[s][t] * decay_com_bat \
                         + E_com_bat_ch[s][t] * eff_com_bat_ch \
                         - E_com_bat_dch[s][t] * (1 / eff_com_bat_dch)
        my_lp_problem += E_com_bat[s][t] <= C_com_bat[s]
        my_lp_problem += E_com_bat[s][t] >= C_com_bat_min * i_com_bat[s]
        my_lp_problem += E_com_bat_ch[s][t] <= C_com_bat[s] * power_eff_com_bat_ch
        my_lp_problem += E_com_bat_dch[s][t] <= C_com_bat[s] * power_eff_com_bat_dch
        # my_lp_problem += i_com_bat_ch[s][t] + i_com_bat_dch[s][t] <= 1
        # my_lp_problem += E_com_bat_ch[s][t] <= i_com_bat_ch[s][t] * C_com_bat_max
        # my_lp_problem += E_com_bat_dch[s][t] <= i_com_bat_dch[s][t] * C_com_bat_max
        # Hydrogen storage
        my_lp_problem += E_com_hyd[s][t + 1] == E_com_hyd[s][t] * decay_com_hyd \
                         + E_com_hyd_ch[s][t] * eff_com_hyd_ch \
                         - E_com_hyd_dch[s][t] * (1 / eff_com_hyd_dch)
        my_lp_problem += E_com_hyd[s][t] <= C_com_hyd[s]
        my_lp_problem += E_com_hyd[s][t] >= C_com_hyd_min * i_com_hyd[s]
        my_lp_problem += E_com_hyd_ch[s][t] <= C_com_elec[s]
        my_lp_problem += E_com_hyd_dch[s][t] <= C_com_fc[s]
        # my_lp_problem += i_com_hyd_ch[s][t] + i_com_hyd_dch[s][t] <= 1
        # my_lp_problem += E_com_hyd_ch[s][t] <= i_com_hyd_ch[s][t] * C_com_elec_max
        # my_lp_problem += E_com_hyd_dch[s][t] <= i_com_hyd_dch[s][t] * C_com_fc_max
        # my_lp_problem += E_com_hyd_dch[s][t] >= i_com_hyd_dch[s][t] * C_com_fc_min
        # Photovoltaics
        my_lp_problem += E_com_pv[s][t] == A_com_pv[s] * dfw[s]['Q_sol'].iloc[t] * eff_com_pv
    # Initial conditions
    my_lp_problem += E_com_bat[s][0] <= E_com_bat[s][H]
    my_lp_problem += E_com_hyd[s][0] <= E_com_hyd[s][H]
    # Sizing
    my_lp_problem += C_com_bat[s] <= C_com_bat_max * i_com_bat[s]
    my_lp_problem += C_com_bat[s] >= C_com_bat_min * i_com_bat[s]
    my_lp_problem += A_com_pv[s] <= A_com_pv_max * i_com_pv[s]
    my_lp_problem += A_com_pv[s] >= A_com_pv_min * i_com_pv[s]
    my_lp_problem += C_com_hyd[s] <= C_com_hyd_max * i_com_hyd[s]
    my_lp_problem += C_com_hyd[s] >= C_com_hyd_min * i_com_hyd[s]
    my_lp_problem += C_com_fc[s] <= C_com_fc_max * i_com_hyd[s]
    my_lp_problem += C_com_fc[s] >= C_com_fc_min * i_com_hyd[s]
    my_lp_problem += C_com_elec[s] <= C_com_elec_max * i_com_hyd[s]
    my_lp_problem += C_com_elec[s] >= C_com_elec_min * i_com_hyd[s]

    # Objective costs
    # Unit investment costs
    for b in buildings:
        my_lp_problem += p_blg_bat[s][b] == inv_lvl_blg_bat * C_blg_bat[s][b] + a_blg_bat * C_blg_bat[s][b] \
                         + b_blg_bat * i_blg_bat[s][b]
        my_lp_problem += p_blg_tes[s][b] == inv_lvl_blg_tes * C_blg_tes[s][b] + b_blg_tes * i_blg_tes[s][b]
        my_lp_problem += p_blg_stc[s][b] == inv_lvl_blg_stc * A_blg_stc[s][b] + a_blg_stc * A_blg_stc[s][b] \
                         + b_blg_stc * i_blg_stc[s][b]
        my_lp_problem += p_blg_pv[s][b] == inv_lvl_blg_pv * A_blg_pv[s][b] + a_blg_pv * A_blg_pv[s][b] \
                         + b_blg_pv * i_blg_pv[s][b]
        my_lp_problem += p_blg_bol[s][b] == inv_lvl_blg_bol * C_blg_bol[s][b] + b_blg_bol * i_blg_bol[s][b]
        my_lp_problem += p_blg_hp[s][b] == inv_lvl_blg_hp * C_blg_hp[s][b] + b_blg_hp * i_blg_hp[s][b]
    my_lp_problem += p_com_bat[s] == inv_lvl_com_bat * a_com_bat * C_com_bat[s] + b_com_bat * i_com_bat[s]
    my_lp_problem += p_com_pv[s] == inv_lvl_com_pv * A_com_pv[s] + a_com_pv * A_com_pv[s] \
                     + b_com_pv * i_com_pv[s]
    my_lp_problem += p_com_hyd[s] == (inv_lvl_com_hyd + a_com_hyd) * C_com_hyd[s] + b_com_hyd * i_com_hyd[s] + \
                     (inv_lvl_com_elec + a_com_elec) * C_com_elec[s] + b_com_elec * i_com_hyd[s] + \
                     (inv_lvl_com_fc + a_com_fc) * C_com_fc[s] + b_com_fc * i_com_hyd[s]

    # Total costs = Operational + Investment costs + Slacks
    my_lp_problem += O_opr[s] == pulp.lpSum(E_hv_in[s][t] * p_elec[s][t] for t in range(H)) \
                     + pulp.lpSum(V_blg_gas[s][b][t] * p_gas[s][t] for t in range(H) for b in buildings)
    my_lp_problem += O_co2[s] == pulp.lpSum(V_blg_gas[s][b][t] for t in range(H) for b in buildings) * p_co2
    my_lp_problem += O_inv[s] == pulp.lpSum(p_blg_bat[s][b] + p_blg_tes[s][b] + p_blg_hp[s][b] + p_blg_bol[s][b]
                                     + p_blg_pv[s][b] + p_blg_stc[s][b] for b in buildings) \
                     + p_com_bat[s] + p_com_hyd[s] + p_com_pv[s]
    my_lp_problem += O_slk[s] == pulp.lpSum(slk_mv_out[s][t] + slk_mv_in[s][t] for t in range(H))* p_slk \
                                    + pulp.lpSum(slk_blg_in[s][b][t] + slk_blg_out[s][b][t] for b in buildings for t in range(H)) * p_slk \
                                    + pulp.lpSum(O_slk_blg[s][b][t] for b in buildings for t in range(H))
    my_lp_problem += O_tot[s] == O_opr[s] + O_inv[s] + O_co2[s] + O_slk[s]

# Non anticipativity constraint
for s1 in range(scenarios):
    for s2 in range(scenarios):
        if s1 != s2:
            my_lp_problem += C_com_bat[s1] == C_com_bat[s2]
            my_lp_problem += A_com_pv[s1] == A_com_pv[s2]
            my_lp_problem += C_com_hyd[s1] == C_com_hyd[s2]
            my_lp_problem += C_com_elec[s1] == C_com_elec[s2]
            my_lp_problem += C_com_fc[s1] == C_com_fc[s2]
            for b in buildings:
                my_lp_problem += C_blg_bat[s1][b] == C_blg_bat[s2][b]
                my_lp_problem += C_blg_tes[s1][b] == C_blg_tes[s2][b]
                my_lp_problem += A_blg_stc[s1][b] == A_blg_stc[s2][b]
                my_lp_problem += A_blg_pv[s1][b] == A_blg_pv[s2][b]
                my_lp_problem += C_blg_bol[s1][b] == C_blg_bol[s2][b]
                my_lp_problem += C_blg_hp[s1][b] == C_blg_hp[s2][b]

# Objective function
my_lp_problem += pulp.lpSum(probabilities.loc[s]*O_tot[s] for s in range(scenarios))

########################################################################################################################
#  Optimization
print('Problem constructed!')
start_time = time.time()
status = my_lp_problem.solve(pulp.apis.GUROBI_CMD(options=[("threads",2), ("NodefileStart", 200)]))
end_time = time.time() - start_time
print(str(pulp.LpStatus[status]) + ' computing time: ' + str(end_time))
print(pulp.LpStatus[status])
if status != 1:
    print("Model not solved, writing lp file.")
    my_lp_problem.writeLP("model_not_solved.lp")


########################################################################################################################
# Results extraction
df_blg_t_res, df_blg_res, df_com_t_res, df_com_res, df_obj_blg_res, df_obj_res = dict(), dict(), dict(), dict(), \
                                                                                 dict(), dict()
for s in range(scenarios):
    df_blg_t_res[s] = dict()
    for b in buildings:
        df_blg_t_res[s][b] = pd.DataFrame(columns=['T_blg', 'O_slk_blg', 'E_blg_bat', 'Q_tes', 'Q_stc', 'Q_sp', 'Q_tes_ch',
                                                   'Q_tes_dch', 'Q_hp', 'COP_hp', 'E_blg_bat_ch', 'E_blg_bat_dch',
                                                   'E_blg_hp', 'E_blg_pv', 'E_blg_in', 'E_blg_out', 'E_blg_load', 'V_blg_gas',
                                                   'slk_blg_in', 'slk_blg_out'])
    df_blg_res[s] = pd.DataFrame(columns=['C_blg_hp', 'C_blg_bat', 'C_blg_tes', 'A_blg_stc', 'A_blg_pv', 'C_blg_bol',
                                          'i_blg_hp', 'i_blg_bat', 'i_blg_tes', 'i_blg_stc', 'i_blg_pv', 'i_blg_bol'])
    df_com_t_res[s] = pd.DataFrame(columns=['E_hv_in', 'E_mv_out', 'E_mv_in', 'E_com_pv', 'E_com_bat', 'E_com_bat_ch',
                                            'E_com_bat_dch', 'E_com_hyd', 'E_com_hyd_ch', 'E_com_hyd_dch',
                                            'slk_mv_out', 'slk_mv_in'])
    df_obj_blg_res[s] = pd.DataFrame(columns=['p_blg_bat', 'p_blg_tes', 'p_blg_hp', 'p_blg_bol', 'p_blg_pv',
                                              'p_blg_stc'])
df_com_res = pd.DataFrame(columns=['A_com_pv', 'C_com_bat', 'C_com_hyd', 'i_com_bat', 'i_com_hyd', 'i_com_pv'])
df_obj_res = pd.DataFrame(columns=['O_tot', 'O_opr', 'O_co2', 'O_inv', 'O_slk',
                                   'p_com_bat', 'p_com_hyd', 'p_com_pv'])

for s in range(scenarios):
    for b in buildings:
        for t in range(H):
            df_blg_t_res[s][b].loc[t, 'E_blg_out'] = pulp.value(E_blg_out[s][b][t])
            df_blg_t_res[s][b].loc[t, 'E_blg_in'] = pulp.value(E_blg_in[s][b][t])
            df_blg_t_res[s][b].loc[t, 'slk_blg_out'] = pulp.value(slk_blg_out[s][b][t])
            df_blg_t_res[s][b].loc[t, 'slk_blg_in'] = pulp.value(slk_blg_in[s][b][t])
            df_blg_t_res[s][b].loc[t, 'E_blg_load'] = dfb[s][b]['E_blg'].iloc[t]
            df_blg_t_res[s][b].loc[t, 'V_blg_gas'] = pulp.value(V_blg_gas[s][b][t])
            df_blg_t_res[s][b].loc[t, 'T_blg'] = pulp.value(T_blg[s][b][t])
            df_blg_t_res[s][b].loc[t, 'O_slk_blg'] = pulp.value(O_slk_blg[s][b][t])
            df_blg_t_res[s][b].loc[t, 'E_blg_bat'] = pulp.value(E_blg_bat[s][b][t])
            df_blg_t_res[s][b].loc[t, 'Q_tes'] = pulp.value(Q_tes[s][b][t])
            df_blg_t_res[s][b].loc[t, 'Q_stc'] = pulp.value(Q_stc[s][b][t])
            df_blg_t_res[s][b].loc[t, 'Q_sp'] = pulp.value(Q_sp[s][b][t])
            df_blg_t_res[s][b].loc[t, 'Q_tes_ch'] = pulp.value(Q_tes_ch[s][b][t])
            df_blg_t_res[s][b].loc[t, 'Q_tes_dch'] = pulp.value(Q_tes_dch[s][b][t])
            df_blg_t_res[s][b].loc[t, 'Q_hp'] = pulp.value(Q_hp[s][b][t])
            df_blg_t_res[s][b].loc[t, 'COP_hp'] = dfb[s][b]['COP_hp'].iloc[t]
            df_blg_t_res[s][b].loc[t, 'E_blg_bat_ch'] = pulp.value(E_blg_bat_ch[s][b][t])
            df_blg_t_res[s][b].loc[t, 'E_blg_bat_dch'] = pulp.value(E_blg_bat_dch[s][b][t])
            df_blg_t_res[s][b].loc[t, 'E_blg_hp'] = pulp.value(E_blg_hp[s][b][t])
        df_blg_res[s].loc[b, 'C_blg_hp'] = pulp.value(C_blg_hp[s][b])
        df_blg_res[s].loc[b, 'C_blg_bat'] = pulp.value(C_blg_bat[s][b])
        df_blg_res[s].loc[b, 'C_blg_tes'] = pulp.value(C_blg_tes[s][b])
        df_blg_res[s].loc[b, 'A_blg_stc'] = pulp.value(A_blg_stc[s][b])
        df_blg_res[s].loc[b, 'A_blg_pv'] = pulp.value(A_blg_pv[s][b])
        df_blg_res[s].loc[b, 'C_blg_bol'] = pulp.value(C_blg_bol[s][b])
        df_blg_res[s].loc[b, 'i_blg_hp'] = pulp.value(i_blg_hp[s][b])
        df_blg_res[s].loc[b, 'i_blg_bat'] = pulp.value(i_blg_bat[s][b])
        df_blg_res[s].loc[b, 'i_blg_tes'] = pulp.value(i_blg_tes[s][b])
        df_blg_res[s].loc[b, 'i_blg_stc'] = pulp.value(i_blg_stc[s][b])
        df_blg_res[s].loc[b, 'i_blg_pv'] = pulp.value(i_blg_pv[s][b])
        df_blg_res[s].loc[b, 'i_blg_bol'] = pulp.value(i_blg_bol[s][b])
        df_obj_blg_res[s].loc[b, 'p_blg_bat'] = pulp.value(p_blg_bat[s][b])
        df_obj_blg_res[s].loc[b, 'p_blg_tes'] = pulp.value(p_blg_tes[s][b])
        df_obj_blg_res[s].loc[b, 'p_blg_hp'] = pulp.value(p_blg_hp[s][b])
        df_obj_blg_res[s].loc[b, 'p_blg_bol'] = pulp.value(p_blg_bol[s][b])
        df_obj_blg_res[s].loc[b, 'p_blg_pv'] = pulp.value(p_blg_pv[s][b])
        df_obj_blg_res[s].loc[b, 'p_blg_stc'] = pulp.value(p_blg_stc[s][b])
    for t in range(H):
        df_com_t_res[s].loc[t, 'E_hv_in'] = pulp.value(E_hv_in[s][t])
        df_com_t_res[s].loc[t, 'E_mv_out'] = pulp.value(E_mv_out[s][t])
        df_com_t_res[s].loc[t, 'E_mv_in'] = pulp.value(E_mv_in[s][t])
        df_com_t_res[s].loc[t, 'slk_mv_in'] = pulp.value(slk_mv_in[s][t])
        df_com_t_res[s].loc[t, 'slk_mv_out'] = pulp.value(slk_mv_out[s][t])
        df_com_t_res[s].loc[t, 'E_com_pv'] = pulp.value(E_com_pv[s][t])
        df_com_t_res[s].loc[t, 'E_com_bat'] = pulp.value(E_com_bat[s][t])
        df_com_t_res[s].loc[t, 'E_com_bat_ch'] = pulp.value(E_com_bat_ch[s][t])
        df_com_t_res[s].loc[t, 'E_com_bat_dch'] = pulp.value(E_com_bat_dch[s][t])
        df_com_t_res[s].loc[t, 'E_com_hyd'] = pulp.value(E_com_hyd[s][t])
        df_com_t_res[s].loc[t, 'E_com_hyd_ch'] = pulp.value(E_com_hyd_ch[s][t])
        df_com_t_res[s].loc[t, 'E_com_hyd_dch'] = pulp.value(E_com_hyd_dch[s][t])
    df_com_res.loc[s, 'A_com_pv'] = pulp.value(A_com_pv[s])
    df_com_res.loc[s, 'C_com_bat'] = pulp.value(C_com_bat[s])
    df_com_res.loc[s, 'C_com_hyd'] = pulp.value(C_com_hyd[s])
    df_com_res.loc[s, 'i_com_pv'] = pulp.value(i_com_pv[s])
    df_com_res.loc[s, 'i_com_bat'] = pulp.value(i_com_bat[s])
    df_com_res.loc[s, 'i_com_hyd'] = pulp.value(i_com_hyd[s])

    df_obj_res.loc[s, 'O_tot'] = pulp.value(O_tot[s])
    df_obj_res.loc[s, 'O_opr'] = pulp.value(O_opr[s])
    df_obj_res.loc[s, 'O_co2'] = pulp.value(O_co2[s])
    df_obj_res.loc[s, 'O_inv'] = pulp.value(O_inv[s])
    df_obj_res.loc[s, 'O_slk'] = pulp.value(O_slk[s])
    df_obj_res.loc[s, 'p_com_bat'] = pulp.value(p_com_bat[s])
    df_obj_res.loc[s, 'p_com_hyd'] = pulp.value(p_com_hyd[s])
    df_obj_res.loc[s, 'p_com_pv'] = pulp.value(p_com_pv[s])


file = path_out + 'ecp_centralized_stoch' + version + '_nbblgs' + str(len(buildings))

# Write
for s in range(scenarios):
    df_blg_res[s].to_csv(file + '_' + str(s) + '_blg.csv')
    df_com_t_res[s].to_csv(file + '_' + str(s) + '_com_t.csv')
    df_obj_blg_res[s].to_csv(file + '_' + str(s) + '_obj_blg.csv')
    for b in buildings:
        df_blg_t_res[s][b].to_csv(file + '_' + str(s) + '_blg_t_' + str(b) + '.csv')
df_obj_res.to_csv(file + '_obj.csv')
df_com_res.to_csv(file + '_com.csv')
