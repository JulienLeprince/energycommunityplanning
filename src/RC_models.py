import pulp
import pandas as pd
from random import gauss
import math

file_RCmodels = 'https://github.com/JulienLeprince/greybrickbuildings/blob/main/data/calibrated_models.csv?raw=true'
df_RC = pd.read_csv(file_RCmodels, index_col='identification_number')

def RCmodel(lp_problem: pulp.LpProblem,
            model_name: str,
            dfw: pd.DataFrame,
            T_blg: pulp.LpVariable.dicts,
            Q_sp: pulp.LpVariable.dicts,
            O_slk_blg: pulp.LpVariable.dicts,
            H: int,
            b: (str, int),
            s: int = 0,
            T_set: int = 20,
            upsampling_factor: int = 1,
            p_slk: float = 10e3):

    Ti = pulp.LpVariable.dicts('var_Ti_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous')
    Tm = pulp.LpVariable.dicts('var_Tm_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Tm' in model_name else 0
    Te = pulp.LpVariable.dicts('var_Te_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Te' in model_name else 0
    Th = pulp.LpVariable.dicts('var_Th_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Th' in model_name else 0
    Ts = pulp.LpVariable.dicts('var_Ts_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Ts' in model_name else 0

    Qim = pulp.LpVariable.dicts('var_Qim_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')
    Qie = pulp.LpVariable.dicts('var_Qie_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')
    Qih = pulp.LpVariable.dicts('var_Qih_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')
    Qis = pulp.LpVariable.dicts('var_Qis_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')

    s_Ti_pos = pulp.LpVariable.dicts('var_s_Ti_pos_blg'+str(b)+str(s), range(H+1), lowBound=0, cat='Continuous')
    s_Ti_neg = pulp.LpVariable.dicts('var_s_Ti_neg_blg'+str(b)+str(s), range(H+1), lowBound=0, cat='Continuous')
    Q_in = pulp.LpVariable.dicts('var_Qin_blg' + str(b) + str(s), range(H * upsampling_factor), lowBound=0, cat='Continuous')

    var = variance_parameters_identification(model_name, b)

    # Linking constraint between two different sampling times
    for t in range(H):
        lp_problem += T_blg[t + 1] == Ti[t * upsampling_factor + 1] + s_Ti_pos[t + 1] - s_Ti_neg[t + 1]
        lp_problem += O_slk_blg[t] >= (s_Ti_pos[t + 1] + s_Ti_neg[t + 1]) * p_slk
        for i in range(upsampling_factor):
            lp_problem += Q_sp[t] == Q_in[t * upsampling_factor + i]  # * upsampling_factor  (was accounted for twice)
    
    lp_problem += Ti[0] == T_set

    for t in range(H * upsampling_factor):
        t_original_sampling = int(t/upsampling_factor)

        # Sensor
        if 'Ts' in model_name:
            lp_problem += Ts[t+1] - Ts[t] == (Ti[t] - Ts[t])* 1/(df_RC.loc[b, 'Ris']*df_RC.loc[b, 'Cs']*upsampling_factor)
            lp_problem += Qis[t] == (Ts[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Ris'] * df_RC.loc[b, 'Ci']*upsampling_factor)
            if t == 0:
                lp_problem += Ts[t] == T_set
        else:
            lp_problem += Qis[t] == 0

        # Medium
        if 'Tm' in model_name:
            lp_problem += Tm[t+1] - Tm[t] == (Ti[t] - Tm[t])*1/(df_RC.loc[b, 'Rim']*df_RC.loc[b, 'Cm']*upsampling_factor)
                                                #+ gauss(0, var['Tm'])
            lp_problem += Qim[t] == (Tm[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rim'] * df_RC.loc[b, 'Ci']*upsampling_factor)
            if t == 0:
                lp_problem += Tm[t] == T_set
        else:
            lp_problem += Qim[t] == 0

        # Heater
        if 'Th' in model_name:
            lp_problem += Th[t+1] - Th[t] == (Ti[t] - Th[t]) * 1/(df_RC.loc[b, 'Rih'] * df_RC.loc[b, 'Ch']*upsampling_factor) \
                             + Q_in[t]*1/(df_RC.loc[b, 'Ch']*upsampling_factor)
            lp_problem += Qih[t] == (Th[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rih'] * df_RC.loc[b, 'Ci']*upsampling_factor)
            if t == 0:
                lp_problem += Th[t] == T_set
        else:
            lp_problem += Qih[t] == Q_in[t]*1/(df_RC.loc[b, 'Ci']*upsampling_factor)

        # Envelope
        if 'Te' in model_name and 'RiaAe' in model_name:
            lp_problem += Te[t+1] - Te[t] == (Ti[t] - Te[t]) * 1/(df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ce']*upsampling_factor) \
                             + (dfw['T_a'].iloc[t_original_sampling] - Te[t]) * 1/(df_RC.loc[b, 'Rea'] * df_RC.loc[b, 'Ce']*upsampling_factor) \
                             + dfw['Q_sol'].iloc[t_original_sampling]*df_RC.loc[b, 'Ae']/(df_RC.loc[b, 'Ce']*upsampling_factor)
            lp_problem += Qie[t] == (Te[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ci']*upsampling_factor) \
                                        + (dfw['T_a'].iloc[t_original_sampling]-Ti[t]) * 1/(df_RC.loc[b, 'Ria']*df_RC.loc[b, 'Ci']*upsampling_factor)
            if t == 0:
                lp_problem += Te[t] == T_set
        elif 'Te' in model_name:
            lp_problem += Te[t+1] - Te[t] == (Ti[t] - Te[t]) * 1/(df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ce']*upsampling_factor) \
                             + (dfw['T_a'].iloc[t_original_sampling] - Te[t]) * 1/(df_RC.loc[b, 'Rea'] * df_RC.loc[b, 'Ce']*upsampling_factor)
            lp_problem += Qie[t] == (Te[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ci']*upsampling_factor)
            if t == 0:
                lp_problem += Te[t] == T_set
        else:
            lp_problem += Qie[t] == (dfw['T_a'].iloc[t_original_sampling]-Ti[t]) * 1/(df_RC.loc[b, 'Ria']*df_RC.loc[b, 'Ci']*upsampling_factor)

        # Inside temperature
        lp_problem += Ti[t + 1] - Ti[t] == Qie[t] + Qih[t] + Qim[t] + Qis[t] \
                         + dfw['Q_sol'].iloc[t_original_sampling] * df_RC.loc[b, 'Aw'] / (df_RC.loc[b, 'Ci']*upsampling_factor)

    return lp_problem


def variance_parameters_identification(model_name, b):
    """Function to identify the estimated variance per model component in function of the model name.

    The full model name 'TiTmTeThTsAeRia' possesses a component ordering which stays similar across varying model
    orders, allowing the identification of component parameters in function of the model 'length'.
    In another setting, this function would not be necessary: e.g. if the identified model variances had unique names
    pointing to specific components. Here unfortunately pXX parameters refer to different model components depending
    on the model considered and can be identified using the following function.
    Note: p11 always refers to the variance of the principal Ti component."""

    var = {'Tm': 0, 'Te': 0, 'Th': 0, 'Ts': 0}

    if len(model_name) == 4 or len(model_name) == 9:
        comp_1 = model_name[2:4]
        var[comp_1] = math.exp(df_RC.loc[b, 'p22'])
    elif len(model_name) == 6 or len(model_name) == 11:
        comp_1 = model_name[2:4]
        comp_2 = model_name[4:6]
        var[comp_1] = math.exp(df_RC.loc[b, 'p22'])
        var[comp_2] = math.exp(df_RC.loc[b, 'p33'])
    elif len(model_name) == 8 or len(model_name) == 13:
        comp_1 = model_name[2:4]
        comp_2 = model_name[4:6]
        comp_3 = model_name[6:8]
        var[comp_1] = math.exp(df_RC.loc[b, 'p22'])
        var[comp_2] = math.exp(df_RC.loc[b, 'p33'])
        var[comp_3] = math.exp(df_RC.loc[b, 'p44'])
    elif len(model_name) == 10 or len(model_name) == 15:
        comp_1 = model_name[2:4]
        comp_2 = model_name[4:6]
        comp_3 = model_name[6:8]
        comp_4 = model_name[8:10]
        var[comp_1] = math.exp(df_RC.loc[b, 'p22'])
        var[comp_2] = math.exp(df_RC.loc[b, 'p33'])
        var[comp_3] = math.exp(df_RC.loc[b, 'p44'])
        var[comp_4] = math.exp(df_RC.loc[b, 'p55'])

    return var
