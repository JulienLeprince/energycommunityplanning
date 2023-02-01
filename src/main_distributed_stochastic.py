import pulp
import pandas as pd
import numpy as np
import time
import os
from datetime import date

# Loading parameters
from parameters import *
# Loading RC models
from RC_models_15mins import *

# Path definition
# path_in = r'C:\Users\20190285\surfdrive\05_Data\054_inout\0548_ECP\in\scenarios/'
# path_out = r'C:\Users\20190285\surfdrive\05_Data\054_inout\0548_ECP\out/'
folder = 'poc_distributedstochastic_' + str(date.today()) + '/'
path_in = '../data/in/'
path_out = '../data/out/' + folder
path_src = ''
version = 'proofofconcept_5buildings_5scenarios'

# Create folder
if not os.path.exists(path_out):
    os.makedirs(path_out)


# RC building models
file_RCmodels = path_in+'all_greybox_fits.csv'
df_RC = pd.read_csv(file_RCmodels, index_col='uuid')
df_RC.drop('Unnamed: 0', axis=1, inplace=True)
df_RC = df_RC[df_RC['nCPBES'] < 0.01]
df_RC.drop(uuids_heatingdemandtoolarge, inplace=True)
df_RC.drop(uuids_upsamplingtolarge, inplace=True)

# Stochastic scenario definition
probabilities = pd.read_csv(path_in+'scenario_probabilities.csv', usecols=[1])
scenarios = probabilities.shape[0]

# TODO - Reducing size of problem here
scenarios = 5

# Reading input data
dfw, dfb = dict(), dict()
p_elec, p_gas = dict(), dict()
for s in range(scenarios):
    dfw[s] = pd.read_csv(path_in+'scenario_'+ str(s) +'.csv', usecols=[1,2])
    dfw[s].rename(columns = {'Ta':'T_a', 'Ps':'Q_sol'}, inplace = True)
    dfw[s] = dfw[s].round(decimals=4)

    dfb[s] = dict()
    bi = 0
    for b in range(83):
        data = pd.read_csv(path_in+'/scenario_'+ str(s) +'.csv', usecols=[5+bi, 6+bi])
        uuid = data.columns[0].split('_')[0]
        dfb[s][uuid] = data
        dfb[s][uuid].rename(columns = {uuid+'_T_blg_set':'T_blg_set', uuid+'_E_elec':'E_blg'}, inplace = True)
        dfb[s][uuid]['E_blg'] = dfb[s][uuid]['E_blg']/1000  # converting from W to kW
        dfb[s][uuid] = dfb[s][uuid].round(decimals=4)
        bi += 2

    p_elec[s] = pd.read_csv(path_in+'/scenario_'+ str(s) +'.csv', usecols=[3])
    p_elec[s] = p_elec[s]['Day-ahead Price [EUR/kWh]']
    p_gas[s] = pd.read_csv(path_in+'/scenario_'+ str(s) +'.csv', usecols=[4])
    p_gas[s] = p_gas[s]['gas_price [EUR/kWh]']
    p_elec[s] = p_elec[s].round(decimals=4)
    p_gas[s] = p_gas[s].round(decimals=4)

H = p_gas[s].shape[0]
buildings = list(dfb[s].keys())
buildings = [value for value in buildings if value in df_RC.index]


# TODO -Reducing size of problem here
buildings = buildings[0:5]


# Calculating heat pump COP
for s in range(scenarios):
    for b in buildings:
        dfb[s][b]['COP_hp'] = pi_hp_1 * np.exp(pi_hp_2 * (dfb[s][b]['T_blg_set'] - dfw[s]['T_a'])) \
                           + pi_hp_3 * np.exp(pi_hp_4 * (dfb[s][b]['T_blg_set'] - dfw[s]['T_a']))



iterations = range(30)

# Result objects declaration
df_objective_function = pd.DataFrame(columns=buildings, index=iterations)
df_blg_t_res, df_com_t_res, df_com_res, df_obj_blg_res, df_obj_res = dict(), dict(), dict(), dict(), dict()
blg_t_cols = ['T_blg', 'O_slk_blg', 'E_blg_bat', 'Q_tes', 'Q_stc', 'Q_sp', 'Q_tes_ch', 'Q_tes_dch', 'Q_hp', 'COP_hp',
              'E_blg_bat_ch', 'E_blg_bat_dch', 'E_blg_hp', 'E_blg_pv', 'E_blg_in', 'E_blg_out', 'E_blg_load', 'V_blg_gas',
              'slk_blg_in', 'slk_blg_out']
blg_cols = ['C_blg_hp', 'C_blg_bat', 'C_blg_tes', 'A_blg_stc', 'A_blg_pv', 'C_blg_bol', 'i_blg_hp', 'i_blg_bat',
            'i_blg_tes', 'i_blg_stc', 'i_blg_pv', 'i_blg_bol',
            'slk_C_hp_max', 'slk_C_bol_max']
com_t_cols = ['E_hv_in', 'E_mv_out', 'E_mv_in', 'E_com_pv', 'E_com_bat', 'E_com_bat_ch', 'E_com_bat_dch', 'E_com_hyd',
              'E_com_hyd_ch', 'E_com_hyd_dch', 'slk_mv_out', 'slk_mv_in']
obj_blg_cols = ['p_blg_bat', 'p_blg_tes', 'p_blg_hp', 'p_blg_bol', 'p_blg_pv', 'p_blg_stc']
com_cols = ['A_com_pv', 'C_com_bat', 'C_com_hyd', 'i_com_bat', 'i_com_hyd', 'i_com_pv']
obj_cols = ['O_tot', 'O_opr', 'O_co2', 'O_inv', 'O_slk', 'p_com_bat', 'p_com_hyd', 'p_com_pv']

for s in range(scenarios):
    df_blg_t_res[s] = dict()
    for b in buildings:
        df_blg_t_res[s][b] = pd.DataFrame(0, index=range(H), columns=blg_t_cols)
    df_com_t_res[s] = pd.DataFrame(0, index=range(H), columns=com_t_cols)
    df_obj_blg_res[s] = pd.DataFrame(0, index=buildings, columns=obj_blg_cols)
df_blg_res = {s: pd.DataFrame(0, index=buildings, columns=blg_cols) for s in range(scenarios)}
df_com_res = pd.DataFrame(0, index=[0], columns=com_cols)
df_obj_res = pd.DataFrame(0, index=[0], columns=obj_cols)

# Distributed iterations
for i in iterations:

    # DISTRIBUTED PROBLEM PER BUILDING
    for b in buildings:


        # Defining LP problem & variables
        my_lp_problem = pulp.LpProblem('My_LP_Problem', pulp.LpMinimize)

        # Variables - building block
        # Continous
        T_blg = pulp.LpVariable.dicts('var_T_blg', (range(scenarios), range(H + 1)), lowBound=0, cat='Continuous')  # Building inside temperature
        O_slk_blg = pulp.LpVariable.dicts('var_O_slk_blg', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
        E_blg_bat = pulp.LpVariable.dicts('var_E_blg_bat', (range(scenarios), range(H + 1)), cat='Continuous')
        Q_tes = pulp.LpVariable.dicts('var_Q_tes', (range(scenarios), range(H + 1)), lowBound=0, cat='Continuous')
        Q_stc = pulp.LpVariable.dicts('var_Q_stc', (range(scenarios), range(H + 1)), lowBound=0, cat='Continuous')
        Q_sp = pulp.LpVariable.dicts('var_Q_sp', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        Q_tes_ch = pulp.LpVariable.dicts('var_Q_tes_ch', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        Q_tes_dch = pulp.LpVariable.dicts('var_Q_tes_dch', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        Q_hp = pulp.LpVariable.dicts('var_Q_hp', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        Q_bol = pulp.LpVariable.dicts('var_Q_bol', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        E_blg_bat_ch = pulp.LpVariable.dicts('var_E_blg_bat_ch', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        E_blg_bat_dch = pulp.LpVariable.dicts('var_E_blg_bat_dch', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        E_blg_hp = pulp.LpVariable.dicts('var_E_blg_hp', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        E_blg_pv = pulp.LpVariable.dicts('var_E_blg_pv', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        E_blg_in = pulp.LpVariable.dicts('var_E_blg_in', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        E_blg_out = pulp.LpVariable.dicts('var_E_blg_out', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        slk_blg_in = pulp.LpVariable.dicts('var_slk_blg_in', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        slk_blg_out = pulp.LpVariable.dicts('var_slk_blg_out', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        V_blg_gas = pulp.LpVariable.dicts('var_V_blg_gas', (range(scenarios), range(H + 1)), lowBound=0, cat='Continous')
        # Sizing
        C_blg_hp = pulp.LpVariable.dicts('var_C_blg_hp', (range(scenarios)), lowBound=0, upBound=C_blg_hp_max, cat='Continous')
        C_blg_bat = pulp.LpVariable.dicts('var_C_blg_bat', (range(scenarios)), lowBound=0, upBound=C_blg_bat_max, cat='Continous')
        C_blg_tes = pulp.LpVariable.dicts('var_C_blg_tes', (range(scenarios)), lowBound=0, upBound=C_blg_tes_max, cat='Continous')
        A_blg_stc = pulp.LpVariable.dicts('var_A_blg_stc', (range(scenarios)), lowBound=0, upBound=A_blg_stc_max, cat='Continous')
        A_blg_pv = pulp.LpVariable.dicts('var_A_blg_pv', (range(scenarios)), lowBound=0, upBound=A_blg_pv_max, cat='Continous')
        C_blg_bol = pulp.LpVariable.dicts('var_C_blg_bol', (range(scenarios)), lowBound=0, upBound=C_blg_bol_max, cat='Continous')
        i_blg_hp = pulp.LpVariable.dicts('var_i_blg_hp', (range(scenarios)), lowBound=0, cat='Binary')
        i_blg_bat = pulp.LpVariable.dicts('var_i_blg_bat', (range(scenarios)), lowBound=0, cat='Binary')
        i_blg_tes = pulp.LpVariable.dicts('var_i_blg_tes', (range(scenarios)), lowBound=0, cat='Binary')
        i_blg_bol = pulp.LpVariable.dicts('var_i_blg_bol', (range(scenarios)), lowBound=0, cat='Binary')
        i_blg_stc = pulp.LpVariable.dicts('var_i_blg_stc', (range(scenarios)), lowBound=0, cat='Binary')
        i_blg_pv = pulp.LpVariable.dicts('var_i_blg_pv', (range(scenarios)), lowBound=0, cat='Binary')
        slk_C_hp_max = pulp.LpVariable.dicts('var_slk_C_hp_max', (range(scenarios)), lowBound=0, cat='Continous')
        slk_C_bol_max = pulp.LpVariable.dicts('var_slk_C_bol_max', (range(scenarios)), lowBound=0, cat='Continous')
        # # Other
        # i_blg_bat_ch = pulp.LpVariable.dicts('var_i_blg_bat_ch', (range(scenarios), range(H + 1)), lowBound=0, cat='Binary')
        # i_blg_bat_dch = pulp.LpVariable.dicts('var_i_blg_bat_dch', (range(scenarios), range(H + 1)), lowBound=0, cat='Binary')
        # i_blg_tes_ch = pulp.LpVariable.dicts('var_i_blg_tes_ch', (range(scenarios), range(H + 1)), lowBound=0, cat='Binary')
        # i_blg_tes_dch = pulp.LpVariable.dicts('var_i_blg_tes_dch', (range(scenarios), range(H + 1)), lowBound=0, cat='Binary')

        # Variables community
        # Continous
        E_hv_in = pulp.LpVariable.dicts('var_E_net', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
        E_mv_out = pulp.LpVariable.dicts('var_E_mv_out', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
        E_mv_in = pulp.LpVariable.dicts('var_E_mv_in', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
        slk_mv_out = pulp.LpVariable.dicts('var_slk_mv_out', (range(scenarios), range(H)), lowBound=0, cat='Continuous')
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
        p_blg_bat = pulp.LpVariable.dicts('var_p_blg_bat', (range(scenarios)), lowBound=0, cat='Continous')
        p_blg_tes = pulp.LpVariable.dicts('var_p_blg_tes', (range(scenarios)), lowBound=0, cat='Continous')
        p_blg_hp = pulp.LpVariable.dicts('var_p_blg_hp', (range(scenarios)), lowBound=0, cat='Continous')
        p_blg_bol = pulp.LpVariable.dicts('var_p_blg_bol', (range(scenarios)), lowBound=0, cat='Continous')
        p_blg_pv = pulp.LpVariable.dicts('var_p_blg_pv', (range(scenarios)), lowBound=0, cat='Continous')
        p_blg_stc = pulp.LpVariable.dicts('var_p_blg_stc', (range(scenarios)), lowBound=0, cat='Continous')
        p_com_bat = pulp.LpVariable.dicts('var_p_com_bat', range(scenarios), lowBound=0, cat='Continous')
        p_com_hyd = pulp.LpVariable.dicts('var_p_com_hyd', range(scenarios), lowBound=0, cat='Continous')
        p_com_pv = pulp.LpVariable.dicts('var_p_com_pv', range(scenarios), lowBound=0, cat='Continous')

        # System constraints
        for s in range(scenarios):

            # Building block
            my_lp_problem = RCmodel(my_lp_problem, df_RC.loc[b, 'model_name'], dfw[s], T_blg[s], Q_sp[s], O_slk_blg[s],
                                    H, b, s, T_set=dfb[s][b]['T_blg_set'].iloc[0])

            for t in range(H):
                # my_lp_problem += T_blg[s][t+1] <= dfb[s][b]['T_blg_set'].iloc[t+1] + T_blg_buffer  # cooling boundary
                my_lp_problem += T_blg[s][t] >= dfb[s][b]['T_blg_set'].iloc[t] - T_blg_buffer  # heating boundary
                # Battery
                my_lp_problem += E_blg_bat[s][t + 1] == E_blg_bat[s][t] * decay_blg_bat \
                                 + E_blg_bat_ch[s][t] * eff_blg_bat_ch \
                                 - E_blg_bat_dch[s][t] * (1 / eff_blg_bat_dch)
                my_lp_problem += E_blg_bat[s][t] <= C_blg_bat[s]
                my_lp_problem += E_blg_bat[s][t] >= C_blg_bat_min * i_blg_bat[s]
                my_lp_problem += E_blg_bat_ch[s][t] <= C_blg_bat[s] * power_eff_blg_bat_ch
                my_lp_problem += E_blg_bat_dch[s][t] <= C_blg_bat[s] * power_eff_blg_bat_dch
                # my_lp_problem += i_blg_bat_ch[s][t] + i_blg_bat_dch[s][t] <= 1
                # my_lp_problem += E_blg_bat_ch[s][t] <= i_blg_bat_ch[s][t] * C_blg_bat_max
                # my_lp_problem += E_blg_bat_dch[s][t] <= i_blg_bat_dch[s][t] * C_blg_bat_max
                # Thermal energy storage
                my_lp_problem += Q_tes[s][t + 1] == Q_tes[s][t] * decay_blg_tes \
                                 + Q_tes_ch[s][t] * eff_blg_tes_ch \
                                 - Q_tes_dch[s][t] * (1 / eff_blg_tes_dch)
                my_lp_problem += Q_tes[s][t] <= C_blg_tes[s]
                my_lp_problem += Q_tes_ch[s][t] <= C_blg_tes[s] * power_eff_blg_tes_ch
                my_lp_problem += Q_tes_dch[s][t] <= C_blg_tes[s] * power_eff_blg_tes_dch
                # my_lp_problem += i_blg_tes_ch[s][t] + i_blg_tes_dch[s][t] <= 1
                # my_lp_problem += Q_tes_ch[s][t] <= i_blg_tes_ch[s][t] * C_blg_tes_max
                # my_lp_problem += Q_tes_dch[s][t] <= i_blg_tes_dch[s][t] * C_blg_tes_max
                # Heat pump
                my_lp_problem += Q_hp[s][t] == E_blg_hp[s][t] * dfb[s][b]['COP_hp'].iloc[t]
                my_lp_problem += Q_hp[s][t] <= C_blg_hp[s] + slk_C_hp_max[s]
                # Boiler
                my_lp_problem += Q_bol[s][t] == V_blg_gas[s][t] * eff_blg_bol
                my_lp_problem += Q_bol[s][t] <= C_blg_bol[s] + slk_C_bol_max[s]
                # Photovoltaics
                my_lp_problem += E_blg_pv[s][t] == A_blg_pv[s] * dfw[s]['Q_sol'].iloc[t] * eff_blg_pv
                # Solar thermal collector
                my_lp_problem += Q_stc[s][t] == A_blg_stc[s] * eff_blg_stc * (dfw[s]['Q_sol'].iloc[t]
                                                                                    - U_blg_stc * (T_stc - dfw[s]['T_a'].iloc[t]))
                # Energy balance
                my_lp_problem += Q_sp[s][t] + Q_tes_ch[s][t] == Q_tes_dch[s][t] + Q_hp[s][t] + Q_bol[s][t]
                my_lp_problem += dfb[s][b]['E_blg'].iloc[t] + E_blg_bat_ch[s][t] + E_blg_hp[s][t] + E_blg_out[s][t] \
                                 == E_blg_bat_dch[s][t] + E_blg_pv[s][t] + E_blg_in[s][t]
                # Building slk
                my_lp_problem += E_blg_in[s][t] <= E_lv_max + slk_blg_in[s][t]
                my_lp_problem += E_blg_out[s][t] <= E_lv_max + slk_blg_out[s][t]
            # Sizing
            my_lp_problem += C_blg_bat[s] <= C_blg_bat_max * i_blg_bat[s]
            my_lp_problem += C_blg_bat[s] >= C_blg_bat_min * i_blg_bat[s]
            my_lp_problem += C_blg_tes[s] <= C_blg_tes_max * i_blg_tes[s]
            my_lp_problem += C_blg_hp[s] <= C_blg_hp_max * i_blg_hp[s]
            my_lp_problem += C_blg_hp[s] >= C_blg_hp_min * i_blg_hp[s]
            my_lp_problem += C_blg_bol[s] <= C_blg_bol_max * i_blg_bol[s]
            my_lp_problem += C_blg_bol[s] >= C_blg_bol_min * i_blg_bol[s]
            my_lp_problem += A_blg_pv[s] <= A_blg_pv_max * i_blg_pv[s]
            my_lp_problem += A_blg_pv[s] >= A_blg_pv_min * i_blg_pv[s]
            my_lp_problem += A_blg_stc[s] <= A_blg_stc_max * i_blg_stc[s]
            my_lp_problem += A_blg_stc[s] >= A_blg_stc_min * i_blg_stc[s]
            my_lp_problem += A_blg_pv[s] + A_blg_stc[s] <= A_blg_roof_max
            # Initial conditions
            my_lp_problem += T_blg[s][0] == dfb[s][b]['T_blg_set'].iloc[0]
            my_lp_problem += E_blg_bat[s][0] <= E_blg_bat[s][H]
            my_lp_problem += Q_tes[s][0] <= Q_tes[s][H]
            for t in range(H):
                # Grid topology - energy balance - distributed problem linking constraint
                my_lp_problem += sum(df_blg_t_res[s][bi].loc[t, 'E_blg_out'] for bi in buildings if bi != b) \
                                 + E_blg_out[s][t] + E_mv_out[s][t] \
                                 == sum(df_blg_t_res[s][bi].loc[t, 'E_blg_in'] for bi in buildings if bi != b) \
                                 + E_blg_in[s][t] + E_mv_in[s][t]

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
            my_lp_problem += p_blg_bat[s] == inv_lvl_blg_bat * C_blg_bat[s] + a_blg_bat * C_blg_bat[s] \
                             + b_blg_bat * i_blg_bat[s]
            my_lp_problem += p_blg_tes[s] == inv_lvl_blg_tes * C_blg_tes[s] + b_blg_tes * i_blg_tes[s]
            my_lp_problem += p_blg_stc[s] == inv_lvl_blg_stc * A_blg_stc[s] + a_blg_stc * A_blg_stc[s] \
                             + b_blg_stc * i_blg_stc[s]
            my_lp_problem += p_blg_pv[s] == inv_lvl_blg_pv * A_blg_pv[s] + a_blg_pv * A_blg_pv[s] \
                             + b_blg_pv * i_blg_pv[s]
            my_lp_problem += p_blg_bol[s] == inv_lvl_blg_bol * C_blg_bol[s] + b_blg_bol * i_blg_bol[s]
            my_lp_problem += p_blg_hp[s] == inv_lvl_blg_hp * C_blg_hp[s] + b_blg_hp * i_blg_hp[s]
            my_lp_problem += p_com_bat[s] == inv_lvl_com_bat * a_com_bat * C_com_bat[s] + b_com_bat * i_com_bat[s]
            my_lp_problem += p_com_pv[s] == inv_lvl_com_pv * A_com_pv[s] + a_com_pv * A_com_pv[s] \
                             + b_com_pv * i_com_pv[s]
            my_lp_problem += p_com_hyd[s] == (inv_lvl_com_hyd + a_com_hyd) * C_com_hyd[s] + b_com_hyd * i_com_hyd[s] + \
                             (inv_lvl_com_elec + a_com_elec) * C_com_elec[s] + b_com_elec * i_com_hyd[s] + \
                             (inv_lvl_com_fc + a_com_fc) * C_com_fc[s] + b_com_fc * i_com_hyd[s]

            # Total costs = Operational + Investment costs
            my_lp_problem += O_opr[s] == pulp.lpSum(E_hv_in[s][t] * p_elec[s][t] for t in range(H)) \
                             + pulp.lpSum(V_blg_gas[s][t] * p_gas[s][t] for t in range(H))
            my_lp_problem += O_co2[s] == pulp.lpSum(V_blg_gas[s][t] for t in range(H)) * p_co2
            my_lp_problem += O_inv[s] == p_blg_bat[s] + p_blg_tes[s] + p_blg_hp[s] + p_blg_bol[s] + p_blg_pv[s] \
                             + p_blg_stc[s] + p_com_bat[s] + p_com_hyd[s] + p_com_pv[s]
            my_lp_problem += O_slk[s] == pulp.lpSum(slk_mv_out[s][t] + slk_mv_in[s][t] for t in range(H)) * p_slk \
                             + pulp.lpSum(slk_blg_in[s][t] + slk_blg_out[s][t] for t in range(H)) * p_slk \
                             + pulp.lpSum(O_slk_blg[s][t] for t in range(H)) \
                             + pulp.lpSum(slk_C_hp_max[s] + slk_C_bol_max[s]) * p_C_slk
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
                    my_lp_problem += C_blg_bat[s1] == C_blg_bat[s2]
                    my_lp_problem += C_blg_tes[s1] == C_blg_tes[s2]
                    my_lp_problem += A_blg_stc[s1] == A_blg_stc[s2]
                    my_lp_problem += A_blg_pv[s1] == A_blg_pv[s2]
                    my_lp_problem += C_blg_bol[s1] == C_blg_bol[s2]
                    my_lp_problem += C_blg_hp[s1] == C_blg_hp[s2]

        # Objective function
        my_lp_problem += pulp.lpSum(probabilities.iloc[s]*O_tot[s] for s in range(scenarios))

        ########################################################################################################################
        #  Optimization
        print('Problem constructed!')
        start_time = time.time()
        status = my_lp_problem.solve(pulp.apis.GUROBI_CMD(options=[("threads",2), ("NodefileStart", 200)]))
        end_time = time.time() - start_time
        print(str(pulp.LpStatus[status]) + ' computing time: ' + str(end_time))


        ########################################################################################################################
        # Results extraction
        for s in range(scenarios):
            for t in range(H):
                df_blg_t_res[s][b].loc[t, 'E_blg_out'] = pulp.value(E_blg_out[s][t])
                df_blg_t_res[s][b].loc[t, 'E_blg_in'] = pulp.value(E_blg_in[s][t])
                df_blg_t_res[s][b].loc[t, 'slk_blg_out'] = pulp.value(slk_blg_out[s][t])
                df_blg_t_res[s][b].loc[t, 'slk_blg_in'] = pulp.value(slk_blg_in[s][t])
                df_blg_t_res[s][b].loc[t, 'E_blg_load'] = dfb[s][b]['E_blg'].iloc[t]
                df_blg_t_res[s][b].loc[t, 'V_blg_gas'] = pulp.value(V_blg_gas[s][t])
                df_blg_t_res[s][b].loc[t, 'T_blg'] = pulp.value(T_blg[s][t])
                df_blg_t_res[s][b].loc[t, 'O_slk_blg'] = pulp.value(O_slk_blg[s][t])
                df_blg_t_res[s][b].loc[t, 'E_blg_bat'] = pulp.value(E_blg_bat[s][t])
                df_blg_t_res[s][b].loc[t, 'Q_tes'] = pulp.value(Q_tes[s][t])
                df_blg_t_res[s][b].loc[t, 'Q_stc'] = pulp.value(Q_stc[s][t])
                df_blg_t_res[s][b].loc[t, 'Q_sp'] = pulp.value(Q_sp[s][t])
                df_blg_t_res[s][b].loc[t, 'Q_tes_ch'] = pulp.value(Q_tes_ch[s][t])
                df_blg_t_res[s][b].loc[t, 'Q_tes_dch'] = pulp.value(Q_tes_dch[s][t])
                df_blg_t_res[s][b].loc[t, 'Q_hp'] = pulp.value(Q_hp[s][t])
                df_blg_t_res[s][b].loc[t, 'COP_hp'] = dfb[s][b]['COP_hp'].iloc[t]
                df_blg_t_res[s][b].loc[t, 'E_blg_bat_ch'] = pulp.value(E_blg_bat_ch[s][t])
                df_blg_t_res[s][b].loc[t, 'E_blg_bat_dch'] = pulp.value(E_blg_bat_dch[s][t])
                df_blg_t_res[s][b].loc[t, 'E_blg_hp'] = pulp.value(E_blg_hp[s][t])
            df_blg_res[s].loc[b, 'C_blg_hp'] = pulp.value(C_blg_hp[s])
            df_blg_res[s].loc[b, 'C_blg_bat'] = pulp.value(C_blg_bat[s])
            df_blg_res[s].loc[b, 'C_blg_tes'] = pulp.value(C_blg_tes[s])
            df_blg_res[s].loc[b, 'A_blg_stc'] = pulp.value(A_blg_stc[s])
            df_blg_res[s].loc[b, 'A_blg_pv'] = pulp.value(A_blg_pv[s])
            df_blg_res[s].loc[b, 'C_blg_bol'] = pulp.value(C_blg_bol[s])
            df_blg_res[s].loc[b, 'i_blg_hp'] = pulp.value(i_blg_hp[s])
            df_blg_res[s].loc[b, 'i_blg_bat'] = pulp.value(i_blg_bat[s])
            df_blg_res[s].loc[b, 'i_blg_tes'] = pulp.value(i_blg_tes[s])
            df_blg_res[s].loc[b, 'i_blg_stc'] = pulp.value(i_blg_stc[s])
            df_blg_res[s].loc[b, 'i_blg_pv'] = pulp.value(i_blg_pv[s])
            df_blg_res[s].loc[b, 'i_blg_bol'] = pulp.value(i_blg_bol[s])
            df_obj_blg_res[s].loc[b, 'p_blg_bat'] = pulp.value(p_blg_bat[s])
            df_obj_blg_res[s].loc[b, 'p_blg_tes'] = pulp.value(p_blg_tes[s])
            df_obj_blg_res[s].loc[b, 'p_blg_hp'] = pulp.value(p_blg_hp[s])
            df_obj_blg_res[s].loc[b, 'p_blg_bol'] = pulp.value(p_blg_bol[s])
            df_obj_blg_res[s].loc[b, 'p_blg_pv'] = pulp.value(p_blg_pv[s])
            df_obj_blg_res[s].loc[b, 'p_blg_stc'] = pulp.value(p_blg_stc[s])
            df_blg_res[s].loc[b, 'slk_C_hp_max'] = pulp.value(slk_C_hp_max[s])
            df_blg_res[s].loc[b, 'slk_C_bol_max'] = pulp.value(slk_C_bol_max[s])
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

        # Calculating total objective function
        O_buildings = []
        for s in range(scenarios):
            O_buildings_per_scenario = []
            for bi in buildings:
                if bi != b:
                    O_inv_blg = df_blg_res[s].loc[bi, 'p_blg_bat'] + df_blg_res[s].loc[bi, 'p_blg_tes'] + df_blg_res[s].loc[bi, 'p_blg_stc'] + df_blg_res[s].loc[bi, 'p_blg_pv'] \
                                + df_blg_res[s].loc[bi, 'p_blg_bol'] + df_blg_res[s].loc[bi, 'p_blg_hp']
                    O_co2_blg = np.sum(df_blg_t_res[s][bi].loc[t, 'V_blg_gas']*p_co2 + df_blg_t_res[s][bi].loc[t, 'V_blg_gas'] * p_gas[s][t] for t in range(H))
                    O_slk = np.sum(df_blg_t_res[s][bi].loc[t, 'slk_blg_out']*p_slk + df_blg_t_res[s][bi].loc[t, 'slk_blg_in']*p_slk + df_blg_t_res[s][bi].loc[t, 'O_slk_blg'] for t in range(H)) \
                            + (df_blg_res[s].loc[bi, 'slk_C_hp_max'] + df_blg_res[s].loc[bi, 'slk_C_bol_max']) * p_C_slk
                    O_buildings_per_scenario.append(O_inv_blg + O_co2_blg + O_slk)
            expected_costs = np.sum(O_buildings_per_scenario)*probabilities.iloc[s]
            O_buildings.append(expected_costs)
        tot_objective = pulp.value(my_lp_problem.objective) + np.sum(O_buildings)
        df_objective_function.loc[i, b] = tot_objective


file = path_out + 'ecp_stoch_' + version + '_blgs' + str(len(buildings))
# Write
for s in range(scenarios):
    df_blg_res[s].to_csv(file + '_' + str(s) + '_blg.csv')
    df_com_t_res[s].to_csv(file + '_' + str(s) + '_com_t.csv')
    df_obj_blg_res[s].to_csv(file + '_' + str(s) + '_obj_blg.csv')
    for b in buildings:
        df_blg_t_res[s][b].to_csv(file + '_' + str(s) + '_blg_t_' + str(b) + '.csv')
df_obj_res.to_csv(file + '_obj.csv')
df_com_res.to_csv(file + '_com.csv')
df_objective_function.to_csv(file + '_problem_convergence.csv')
