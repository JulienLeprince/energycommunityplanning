import pulp
import pandas as pd
from random import gauss
import math

path_RCmodels = '../data/in/hourly/'
# path_RCmodels = r'C:\Users\20190285\surfdrive\05_Data\054_inout\0548_ECP\in\scenarios/'
file = 'hourly_greybox_fits.csv'
df_RC = pd.read_csv(path_RCmodels+file, index_col='uuid')

# RC models requiring either heating demands or upsampling too large for the problem
uuids_heatingdemandtoolarge = ['d7126288-4c5a-43dd-ae64-43b0a305aee5']
uuids_upsamplingtolarge = ['026c83b7-3738-49ed-99ba-923c6c627e36',
                           '0ac95da1-229d-49e5-8220-748fffdf686e',
                           '0aec519f-7dca-44bd-b6f8-752c09f2760d',
                           '1a609d80-2d97-440e-b424-967119869026',
                           '1ba99e71-a4ca-4954-b7aa-1c2f124b861f',
                           '1f218889-a8e9-4d08-b16c-5ab6dd4d9a46',
                           '2304f00b-0df5-46eb-9a49-bd6f6b3ee5d2',
                           '27485bd6-4b94-436d-b869-d6caca0ca54b',
                           '2f769914-bc9f-4c13-bc6b-df9ac9c29ddf',
                           '2fb12140-caa2-4256-9ebb-d1ec80a0d797',
                           '353cf99d-a608-41ac-88aa-58d91303f353',
                           '386e088c-5a2f-40c9-ba86-941a598cefed',
                           '3e54f0eb-288f-40a3-a54d-2d8b2308132e',
                           '40330809-e23f-45d2-9427-950ad91f1067',
                           '4073ce6b-66f4-481e-afb1-db2a639f7f75',
                           '4564ae90-6f18-42d7-bf10-02e4a880545b',
                           '46a97d04-5b9d-411b-9c96-0273090ecf4f',
                           '4cd2c433-b4ac-4aa6-b4b3-c3abb51e3fc4',
                           '61977e34-17c8-4773-a331-a4ec25329143',
                           '6bd9053f-3745-444f-92eb-0ddfe0b9ae4f',
                           '6c83c2ca-9d39-4c51-8e0d-1059e0bd512d',
                           '6cea5cb5-c130-4e31-a37c-fb702299d92b',
                           '72eb67b3-1927-4316-abcc-65e0f2c6230f',
                           '7367ce18-a1ac-4c9d-b93b-8565396d8a44',
                           '73f7440c-095e-41d3-b3ed-9da45ffe8d29',
                           '875b6a38-97fa-422e-8a1b-e6fba1fac61f',
                           '8b2665e8-138f-44b9-b331-8ec971db7466',
                           '8bb652ef-4aad-49ec-98ef-cb0c4c300e43',
                           '9583cade-be49-41d5-bedf-97218920f7a8',
                           '9f8cb82d-4fce-4f67-9a94-dd6706a0f257',
                           'a14e39c7-2162-4ec1-8c7f-b2f3c3357c02',
                           'a52aca4f-3b5d-4781-9549-a1785f2f4e7d',
                           'a5dd7a53-4250-4a81-ba9f-4dcf4e3092bd',
                           'a83ae9a4-6fea-431a-8123-b984a25d1f5d',
                           'a8b96f4b-da3c-4d7b-8d3c-54d69f00a39b',
                           'b468f7e3-1adb-4ad7-a466-2f6839e7d975',
                           'b925ff8d-1f8e-44fc-b11f-29836d03c044',
                           'bf13aeff-4292-48a5-b0f9-048aa5a3f2b3',
                           'c1884ab5-6d64-48ac-ada7-fe4908f004ba',
                           'c9449c3e-5990-426b-944d-a7f315445201',
                           'c9a722a0-fb32-403f-812d-462fd19d4c1b',
                           'caf25686-dc1e-404f-abfa-aa223a9eb14d',
                           'cbd6a264-f8bc-42b1-b3b5-85ba01fca6a4',
                           'cc3342e3-554c-41b1-941b-f2d6d45c4205',
                           'cc8f0bb6-c2d4-4e75-b910-a1003d1aa8c1',
                           'd1a6575d-e903-4be9-a824-f5c63a747edf',
                           'd1c40b63-8f44-43a4-9fb5-bfb2f3f2b443',
                           'd3edcfbe-4ad6-42de-8e7d-ca10330e9e0b',
                           'd9342075-0ced-4913-9475-33c2bfce494f',
                           'dc0d61bd-814c-4452-a95f-33db4aff7e03',
                           'e249d2f7-a214-4811-9358-c77d0c56901f',
                           'e3c6809f-74a2-4d61-af25-c9d49d70cb07',
                           'e7a0f133-ad9b-4ca5-ad9d-080ad8654bfc',
                           'e80ecc6a-ec19-40bc-a688-8b93c3845b7a',
                           'eb1c78cc-e7cb-4e83-8ec5-b5592c737a87',
                           'eb7064fa-d0fd-4a7a-aaed-c8852c3f088a',
                           'ed602d72-d554-4820-8629-81ea51fb4ac0',
                           'ee716447-565b-428d-be11-595ecf065632',
                           'ef0e61b7-76c3-46ef-abe8-c90be8ffe90b',
                           'ef5765ec-d47e-41a7-b352-973a02c9a683',
                           'f3b63104-c9bc-49bd-b5ee-6a8d2389905f',
                           'f4274906-4b55-4363-9499-8f8b02051e31',
                           'fb345df7-24e8-4941-9e74-0bd86c42de00',
                           'fbf68c28-8e56-485e-8afe-7582e88c6324',
                           'fd4f8fd7-46e6-4ad5-bb73-9e89aee25a9f']


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
            p_slk: int = 10e3):

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
