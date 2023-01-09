import pulp
import pandas as pd
from random import gauss
import math

# path_RCmodels = r'C:/energycommunityplanning/data/in/'
path_RCmodels = '../data/in/'
file = 'all_greybox_fits.csv'
df_RC = pd.read_csv(path_RCmodels+file, index_col='uuid')


def RCmodel(lp_problem: pulp.LpProblem,
            model_name: str,
            dfw: pd.DataFrame,
            T_blg: pulp.LpVariable.dicts,
            Q_sp: pulp.LpVariable.dicts,
            H: int,
            b,
            s: int = 0):

    Ti = pulp.LpVariable.dicts('var_Ti_blg'+str(b)+str(s), range(H + 1), cat='Continuous')
    Tm = pulp.LpVariable.dicts('var_Tm_blg'+str(b)+str(s), range(H + 1), cat='Continuous') if 'Tm' in model_name else 0
    Te = pulp.LpVariable.dicts('var_Te_blg'+str(b)+str(s), range(H + 1), cat='Continuous') if 'Te' in model_name else 0
    Th = pulp.LpVariable.dicts('var_Th_blg'+str(b)+str(s), range(H + 1), cat='Continuous') if 'Th' in model_name else 0
    Ts = pulp.LpVariable.dicts('var_Ts_blg'+str(b)+str(s), range(H + 1), cat='Continuous') if 'Ts' in model_name else 0

    Qim = pulp.LpVariable.dicts('var_Qim_blg'+str(b)+str(s), range(H), cat='Continuous')
    Qie = pulp.LpVariable.dicts('var_Qie_blg'+str(b)+str(s), range(H), cat='Continuous')
    Qih = pulp.LpVariable.dicts('var_Qih_blg'+str(b)+str(s), range(H), cat='Continuous')
    Qis = pulp.LpVariable.dicts('var_Qis_blg'+str(b)+str(s), range(H), cat='Continuous')

    var = variance_parameters_identification(model_name, b)

    for t in range(H):

        # Sensor
        if 'Ts' in model_name:
            lp_problem += Ts[t+1] - Ts[t] == (Ti[t] - Ts[t])* 1/(df_RC.loc[b, 'Ris']*df_RC.loc[b, 'Cs']) #+ gauss(0, var['Ts'])
            lp_problem += T_blg[b][t+1] == Ts[t+1] #+ gauss(0, math.exp(df_RC.loc[b, 'e11']))
            lp_problem += Qis[t] == (Ts[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Ris'] * df_RC.loc[b, 'Ci'])
            # if t == 0:
            #     lp_problem += Ts[t] == df_RC.loc[b, 'Ts0']
        else:
            lp_problem += Qis[t] == 0
            lp_problem += T_blg[b][t + 1] == Ti[t + 1] #+ gauss(0, math.exp(df_RC.loc[b, 'e11']))

        # Medium
        if 'Tm' in model_name:
            lp_problem += Tm[t+1] - Tm[t] == (Ti[t] - Tm[t])*1/(df_RC.loc[b, 'Rim']*df_RC.loc[b, 'Cm']) #+ gauss(0, var['Tm'])
            lp_problem += Qim[t] == (Tm[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rim'] * df_RC.loc[b, 'Ci'])
            # if t == 0:
            #     lp_problem += Tm[t] == df_RC.loc[b, 'Tm0']
        else:
            lp_problem += Qim[t] == 0

        # Heater
        if 'Th' in model_name:
            lp_problem += Th[t+1] - Th[t] == (Ti[t] - Th[t]) * 1/(df_RC.loc[b, 'Rih'] * df_RC.loc[b, 'Ch']) \
                             + Q_sp[b][t]*1/(df_RC.loc[b, 'Ch']) #+ gauss(0, var['Th'])
            lp_problem += Qih[t] == (Th[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rih'] * df_RC.loc[b, 'Ci'])
            # if t == 0:
            #     lp_problem += Th[t] == df_RC.loc[b, 'Th0']
        else:
            lp_problem += Qih[t] == Q_sp[b][t]*1/(df_RC.loc[b, 'Ci'])

        # Envelope
        if 'Te' in model_name and 'RiaAe' in model_name:
            lp_problem += Te[t+1] - Te[t] == (Ti[t] - Te[t]) * 1/(df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ce']) \
                             + (dfw['T_a'].iloc[t] - Te[t]) * 1/(df_RC.loc[b, 'Rea'] * df_RC.loc[b, 'Ce']) \
                             + dfw['Q_sol'].iloc[t]*df_RC.loc[b, 'Ae']/df_RC.loc[b, 'Ce']
                             #+ gauss(0, var['Te'])
            lp_problem += Qie[t] == (Te[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ci']) \
                                        + (dfw['T_a'].iloc[t]-Ti[t]) * 1/(df_RC.loc[b, 'Ria']*df_RC.loc[b, 'Ci'])
            # if t == 0:
            #     lp_problem += Te[t] == df_RC.loc[b, 'Te0']
        elif 'Te' in model_name:
            lp_problem += Te[t+1] - Te[t] == (Ti[t] - Te[t]) * 1/(df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ce']) \
                             + (dfw['T_a'].iloc[t] - Te[t]) * 1/(df_RC.loc[b, 'Rea'] * df_RC.loc[b, 'Ce'])
                             #+ gauss(0, var['Te'])
            lp_problem += Qie[t] == (Te[t] - Ti[t]) * 1 / (df_RC.loc[b, 'Rie'] * df_RC.loc[b, 'Ci'])
            # if t == 0:
            #     lp_problem += Te[t] == df_RC.loc[b, 'Te0']
        else:
            lp_problem += Qie[t] == (dfw['T_a'].iloc[t]-Ti[t]) * 1/(df_RC.loc[b, 'Ria']*df_RC.loc[b, 'Ci'])

        # Inside temperature
        lp_problem += Ti[t + 1] - Ti[t] == Qie[t] + Qih[t] + Qim[t] + Qis[t] \
                         + dfw['Q_sol'].iloc[t] * df_RC.loc[b, 'Aw'] / df_RC.loc[b, 'Ci']
                         #+ gauss(0, math.exp(df_RC.loc[b, 'p11']))

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
