import pulp
import pandas as pd
from random import gauss
import math

# path_RCmodels = r'C:/energycommunityplanning/data/in/'
path_RCmodels = '../data/in/'
file = 'all_greybox_fits.csv'
df_RC = pd.read_csv(path_RCmodels+file, index_col='uuid')

# RC models requiring either heating demands or upsampling too large for the problem
uuids_heatingdemandtoolarge = ['12e8ebfa-0ff9-4924-b2e1-d70f47005b2a',
                               '1f218889-a8e9-4d08-b16c-5ab6dd4d9a46',
                               'b8787ac9-4ae7-4e83-bc68-826b74e6c5d6',
                               'b9b35765-804b-4a67-a59b-248b7ce94080',
                               'd1a6575d-e903-4be9-a824-f5c63a747edf',
                               '9f8cb82d-4fce-4f67-9a94-dd6706a0f257']
uuids_upsamplingtolarge = ['17aad2c4-aae9-4489-bee9-bcd7a666bf0c',
                           '28267176-3b1d-454a-ab66-5e6c8205f6db',
                           '2ca893a7-1fb6-4b6b-9c17-490489ff89f7',
                           '3dcd990b-b1a7-4644-a472-8037ab3b15e6',
                           '46a97d04-5b9d-411b-9c96-0273090ecf4f',
                           '50b76bcc-76b5-4c77-971f-ec90e4ab50d2',
                           '59ced9d6-ff85-49ea-8763-694c27fe3479',
                           '5a86896f-6278-4ab1-be4f-3513ae75a4a8',
                           '63bfc952-bf89-4b73-8f44-08c1912bdbb0',
                           '6f792023-c9a4-477f-9181-9fc5c1e7344a',
                           '7367ce18-a1ac-4c9d-b93b-8565396d8a44',
                           '7ac4f875-376b-4dc9-871c-bb057605abd3',
                           '8e32cf10-a7cf-48d1-bb4c-e5edec486981',
                           '95710169-ff43-415f-a317-ebdb3b0a60d1',
                           'a14e39c7-2162-4ec1-8c7f-b2f3c3357c02',
                           'a2975782-144b-4c79-8f78-d55ea3f352ab',
                           'a52aca4f-3b5d-4781-9549-a1785f2f4e7d',
                           'a5dd7a53-4250-4a81-ba9f-4dcf4e3092bd',
                           'a83ae9a4-6fea-431a-8123-b984a25d1f5d',
                           'b3c8cb04-ea5d-458b-89f8-374c4916f16c',
                           'cde5e2a7-c092-48c0-b23b-9ae9b513b34d',
                           'ce7f1464-675c-41d3-97ca-f174b2f29fb8',
                           'd2faad8d-8f55-47d6-93fc-deafe527b287',
                           'd7189d13-171c-43bb-93b0-3686ca8b21a8',
                           'd9342075-0ced-4913-9475-33c2bfce494f',
                           'e9539c2b-2960-4f38-86e4-b6675fe0f977',
                           'f09e2ae9-94c3-4311-b780-67dbf78e1006',
                           'f4bd4805-012e-4ba1-ae60-1397b443aafb',
                           'fbf68c28-8e56-485e-8afe-7582e88c6324',
                           '151effd4-4ffe-464d-ad61-e0667eee90d6']


def RCmodel(lp_problem: pulp.LpProblem,
            model_name: str,
            dfw: pd.DataFrame,
            T_blg: pulp.LpVariable.dicts,
            Q_sp: pulp.LpVariable.dicts,
            H: int,
            b: (str, int),
            s: int = 0,
            T_set: int = 20,
            upsampling_factor: int = 4):

    Ti = pulp.LpVariable.dicts('var_Ti_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous')
    Tm = pulp.LpVariable.dicts('var_Tm_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Tm' in model_name else 0
    Te = pulp.LpVariable.dicts('var_Te_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Te' in model_name else 0
    Th = pulp.LpVariable.dicts('var_Th_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Th' in model_name else 0
    Ts = pulp.LpVariable.dicts('var_Ts_blg'+str(b)+str(s), range(H * upsampling_factor + 1), cat='Continuous') if 'Ts' in model_name else 0

    Qim = pulp.LpVariable.dicts('var_Qim_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')
    Qie = pulp.LpVariable.dicts('var_Qie_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')
    Qih = pulp.LpVariable.dicts('var_Qih_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')
    Qis = pulp.LpVariable.dicts('var_Qis_blg'+str(b)+str(s), range(H * upsampling_factor), cat='Continuous')

    Q_in = pulp.LpVariable.dicts('var_Qin_blg' + str(b) + str(s), range(H * upsampling_factor), cat='Continuous')

    var = variance_parameters_identification(model_name, b)

    # Linking constraint between two different sampling times
    for t in range(H):
        lp_problem += T_blg[t + 1] == Ti[t * upsampling_factor + 1]
        for i in range(upsampling_factor):
            lp_problem += Q_sp[t] == Q_in[t * upsampling_factor + i]*upsampling_factor

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
