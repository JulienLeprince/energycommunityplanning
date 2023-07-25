import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path_gitrepo = r'C:/Users/20190285/Documents/GitHub/energycommunityplanning/'
path_res = path_gitrepo + 'data/out/'
path_fig = path_gitrepo + 'fig/'

# Sensitivity plot
nb_of_scenarios = 10

sa_setups = ['userbehavior', 'climate', 'economic']
sa_nb = [x for x in range(nb_of_scenarios)]

i = 0
# Building design values
for sa_setup in sa_setups:
    for sa_i in sa_nb:
        if sa_setup == 'climate' and sa_i == 9:
            pass
        else:
            blg_files = path_res + 'SA_distrib_totscenarios10_publication_run_nbblgs41_setup_' \
                        + sa_setup + str(sa_i) + '_0_blg.csv'
            df_design = pd.read_csv(blg_files)
            df_design = df_design.rename(columns={'Unnamed: 0': 'building'})
            df_design['uncertainty factor'] = [sa_setup]*len(df_design)
            df_design_all = df_design if i == 0 else pd.concat([df_design_all, df_design], axis=0)
            i += 1

# Community design value
i = 0
for sa_setup in sa_setups:
    for sa_i in sa_nb:
        if sa_setup == 'climate' and sa_i == 9:
            pass
        else:
            comfiles = path_res + 'SA_distrib_totscenarios10_publication_run_nbblgs41_setup_' \
                       + sa_setup + str(sa_i) + '_com.csv'
            df_com = pd.read_csv(comfiles, index_col=[0])
            df_com.rename(index={0: str(sa_i)}, inplace=True)
            df_com['uncertainty factor'] = [sa_setup] * len(df_com)
            df_com_all = df_com if i == 0 else pd.concat([df_com_all, df_com], axis=0)
            i += 1

# Transform building uuid data to numbers
unique_uuids = df_design_all['building'].unique()
anonymized_uuids = [str(x+1) for x in range(len(unique_uuids))]
df_design_all['building'] = df_design_all['building'].replace(unique_uuids, anonymized_uuids)


########################################################################################################################

# Multi plot boiler + heatpump + tes

data = df_design_all #[df_design_all['uncertainty factor'] == 'userbehavior']

sns.set(rc={"figure.figsize": (8, 10)})
sns.set_theme(style="ticks", font='serif', font_scale=.8)  #  whitegrid
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(width_ratios=[6, 1, 1], wspace=0.1))
sns.pointplot(
        data=data, x='C_blg_bol', y="building", hue="uncertainty factor",
        join=False, dodge=.8 - .8 / 3,
        palette='dark', #['blue', 'red'],
        markers="d", scale=.55,
        errwidth=.4,
        ax=ax1
        )
sns.stripplot(
    data=data, x='C_blg_bol', y="building", hue="uncertainty factor",
    dodge=True, alpha=.55,  size=4,
    palette='dark', #['blue', 'red'],
    label='',
    zorder=1, ax=ax1
)
ax1.set_xlabel('Boiler capacity\n[kWh]')
ax1.set_ylabel('Buildings identification number')
ax1.set_xlim(-0.1, 14)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='0.95', linestyle='dashed')
ax1.yaxis.grid(color='0.95', linestyle='dashed')
sns.pointplot(
        data=data, x='C_blg_hp', y="building", hue="uncertainty factor",
        join=False, dodge=.8 - .8 / 3, palette="dark",
        markers="d", scale=.55,
        errwidth=.4,
        ax=ax2
        )
sns.stripplot(
    data=data, x='C_blg_hp', y="building", hue="uncertainty factor",
    dodge=True, alpha=.55,  size=4,
    palette='dark', #['blue', 'red'],
    zorder=1, ax=ax2
)
ax2.set_xlabel('Heat-pump\ncapacity\n[kWh]')
ax2.set_ylabel('')
ax2.set_xlim(-0.1, 7)
ax2.set_axisbelow(True)
ax2.xaxis.grid(color='0.95', linestyle='dashed')
ax2.yaxis.grid(color='0.95', linestyle='dashed')
ax2.legend([], [], frameon=False)
sns.pointplot(
        data=data, x='C_blg_tes', y="building", hue="uncertainty factor",
        join=False, dodge=.8 - .8 / 3, palette="dark",
        markers="d", scale=.55,
        errwidth=.4,
        ax=ax3
        )
sns.stripplot(
    data=data, x='C_blg_tes', y="building", hue="uncertainty factor",
    dodge=True, alpha=.55,  size=4,
    palette='dark', #['blue', 'red'],
    zorder=1, ax=ax3
)
ax3.set_xlabel('Thermal\nenergy\nstorage\ncapacity\n[kWh]')
ax3.set_ylabel('')
ax3.set_xlim(-0.05, 3)
ax3.set_axisbelow(True)
ax3.xaxis.grid(color='0.95', linestyle='dashed')
ax3.yaxis.grid(color='0.95', linestyle='dashed')
ax3.legend([], [], frameon=False)

lns = ax1.lines  #+ ax2.lines + ax3.lines
labs = [l.get_label() for l in lns]
ax1.legend(title='Uncertainty factor')

# Save
plt.savefig(path_fig+'SA_allheatingutilities.pdf', bbox_inches='tight')




########################################################################################################################

# Sensitivity analysis-  multi plot boiler + heatpump + tes + optimal design

# Stochastic optimal design
folder = 'hourly_full_distributedstochastic_2023-02-14/'
blg_files = path_res + folder + 'ecp_distributed_stoch_totscenarios10_publication_run_nbblgs41_0_blg.csv'
df_design = pd.read_csv(blg_files, index_col=0)
for col in df_design.columns:
    if 'i_' in col:
        df_design.drop(col, axis=1, inplace=True)
df_design.reset_index(drop=True, inplace=True)
#df_design.sort_values('C_blg_bol', inplace=True)
df_design.index = df_design.index.astype(str)
df_design.rename(columns={'C_blg_hp': 'heat pump',
                          'C_blg_bol': 'boiler',
                          'C_blg_bat': 'battery',
                          'C_blg_tes': 'thermal energy storage',
                          'A_blg_stc': 'solar thermal collector',
                          'A_blg_pv': 'photovoltaics'},
                 inplace=True)

# plot
sns.set(rc={"figure.figsize": (6.3, 16)})
sns.set_theme(style="ticks", font='serif', font_scale=.8)  #  whitegrid
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(width_ratios=[6, 1, 1], wspace=0.1))
sns.pointplot(
        data=data, x='C_blg_bol', y="building", hue="uncertainty factor",
        join=False, dodge=.8 - .8 / 3,
        palette='dark', #['blue', 'red'],
        markers="d", scale=.55,
        errwidth=.4, zorder=2,
        ax=ax1
        )
sns.stripplot(
    data=data, x='C_blg_bol', y="building", hue="uncertainty factor",
    dodge=True, alpha=.55,  size=4,
    palette='dark', #['blue', 'red'],
    label='',
    zorder=2, ax=ax1
)
sns.scatterplot(
    data=df_design.reset_index(), x='boiler', y="index",
    alpha=.5,
    s=150,
    color='black',
    marker="*",
    label='optimal',
    zorder=1, ax=ax1
)
ax1.set_xlabel('Boiler capacity\n[kWh]')
ax1.set_ylabel('Buildings identification number')
ax1.set_xlim(-0.1, 14.3)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='0.95', linestyle='dashed')
ax1.yaxis.grid(color='0.95', linestyle='dashed')
sns.pointplot(
        data=data, x='C_blg_hp', y="building", hue="uncertainty factor",
        join=False, dodge=.8 - .8 / 3, palette="dark",
        markers="d", scale=.55,
        errwidth=.4,
        zorder=2,
        ax=ax2
        )
sns.stripplot(
    data=data, x='C_blg_hp', y="building", hue="uncertainty factor",
    dodge=True, alpha=.55,  size=4,
    palette='dark', #['blue', 'red'],
    zorder=2, ax=ax2
)
sns.scatterplot(
    data=df_design.reset_index(), x='heat pump', y="index",
    alpha=.5,
    s=150,
    color='black',
    marker="*",
    zorder=1, ax=ax2
)
ax2.set_xlabel('Heat\npump\ncapacity\n[kWh]')
ax2.set_ylabel('')
ax2.set_xlim(-0.1, 7.5)
ax2.set_axisbelow(True)
ax2.xaxis.grid(color='0.95', linestyle='dashed')
ax2.yaxis.grid(color='0.95', linestyle='dashed')
ax2.legend([], [], frameon=False)
sns.pointplot(
        data=data, x='C_blg_tes', y="building", hue="uncertainty factor",
        join=False, dodge=.8 - .8 / 3, palette="dark",
        markers="d", scale=.55,
        errwidth=.4,
        zorder=2,
        ax=ax3
        )
sns.stripplot(
    data=data, x='C_blg_tes', y="building", hue="uncertainty factor",
    dodge=True, alpha=.55,  size=4,
    palette='dark', #['blue', 'red'],
    zorder=2, ax=ax3
)
sns.scatterplot(
    data=df_design.reset_index(), x='thermal energy storage', y="index",
    alpha=.5,
    s=150,
    color='black',
    marker="*",
    zorder=1, ax=ax3
)
ax3.set_xlabel('Thermal\nenergy\nstorage\ncapacity\n[kWh]')
ax3.set_ylabel('')
ax3.set_xlim(-0.05, 3.2)
ax3.set_axisbelow(True)
ax3.xaxis.grid(color='0.95', linestyle='dashed')
ax3.yaxis.grid(color='0.95', linestyle='dashed')
ax3.legend([], [], frameon=False)

# added these three lines
lns = ax1.lines  #+ ax2.lines + ax3.lines
labs = [l.get_label() for l in lns]
ax1.legend(title='Uncertainty factor', loc='lower right')

# Save
plt.savefig(path_fig+'SA_allheatingutilities_woptimal.pdf', bbox_inches='tight')



########################################################################################################################

## Stochastic optimal design
folder = 'hourly_full_distributedstochastic_2023-02-14/'
blg_files = path_res + folder + 'ecp_distributed_stoch_totscenarios10_publication_run_nbblgs41_0_blg.csv'
df_design = pd.read_csv(blg_files, index_col=0)
for col in df_design.columns:
    if 'i_' in col:
        df_design.drop(col, axis=1, inplace=True)
df_design.reset_index(drop=True, inplace=True)
df_design.index = df_design.index + 1
df_design.sort_values('C_blg_bol', inplace=True)
df_design.index = df_design.index.astype(str)
df_design.rename(columns={'C_blg_hp': 'heat pump',
                          'C_blg_bol': 'boiler',
                          'C_blg_bat': 'battery',
                          'C_blg_tes': 'thermal energy storage',
                          'A_blg_stc': 'solar thermal collector',
                          'A_blg_pv': 'photovoltaics'},
                 inplace=True)

cols_capa = ['heat-pump', 'boiler', 'battery', 'thermal energy storage']
cols_area = ['solar thermal collector', 'photovoltaics']

# Plot
sns.set(rc={"figure.figsize": (9, 3)})
sns.set_theme(style="ticks", font='serif', font_scale=.8)
fig, ax1 = plt.subplots()
ax1.yaxis.grid(color='0.95', linestyle='dashed')
sns.lineplot(data=df_design, linewidth=1, zorder=3,
             sort=False,
             markers=True,
             ax=ax1)
ax1.legend(title='Design variables')
ax1.set_xlabel('Buildings')
ax1.set_ylabel('Capacity [kWh] or Area [m$^2$]')

# Save
plt.savefig(path_fig+'stochastic_optimaL_design.pdf', bbox_inches='tight')



########################################################################################################################

# Building inside temp control

# Preprocessing
sa = 0
blg_files = path_res + 'ecp_distributed_stoch_totscenarios5_publication_run_nbblgs41_' + str(sa) + '_blg_t_5.csv'
df_control = pd.read_csv(blg_files, index_col=0)
timestamp = pd.date_range(start='01.01.2019', end='01.01.2020', freq='H')
timestamp = timestamp.drop(timestamp[-1])
df_control.set_index(timestamp, inplace=True)
df_control.rename(columns={'T_blg': '$T_{building}$',
                           'Q_sp': '$Q_{space heating}$',
                           # 'E_blg_in': '$E_{electricity}$',
                           # 'V_blg_gas': '$V_{gas}$',
                           },
                  inplace=True)

path_to_weather = r'C:\Users\20190285\surfdrive\05_Data\054_inout\0548_ECP\in\scenarios_hourly/'
weather_file = path_to_weather + 'scenario_' + str(sa) + '.csv'
df_weather = pd.read_csv(weather_file, index_col=0)
df_weather.set_index(timestamp, inplace=True)
df_weather.rename(columns={'Ta': '$T_{ambiant}$',
                           'Ps': '$Q_{irradiance}$',
                           '5_T_blg_set': '$T_{comfort threshold}$',
                           # 'E_blg_in': '$E_{electricity}$',
                           # 'V_blg_gas': '$V_{gas}$',
                           },
                  inplace=True)
df_weather['$T_{comfort threshold}$'] = df_weather['$T_{comfort threshold}$'] - 1
df_eco = df_weather[['Day-ahead Price [EUR/kWh]', 'gas_price [EUR/kWh]']].copy()
df_eco.rename(columns={'Day-ahead Price [EUR/kWh]': '$p_{electricity}$',
                           'gas_price [EUR/kWh]': '$p_{gas}$',
                           },
              inplace=True)

df_temps = pd.concat([df_control['$T_{building}$'], df_weather[['$T_{ambiant}$','$T_{comfort threshold}$']]], axis=1)
df_heat = pd.concat([df_control['$Q_{space heating}$'], df_weather['$Q_{irradiance}$']], axis=1)



# Plot inside temp control
start_i = 500
end_i = 700
sns.set(rc={"figure.figsize": (7, 4.5)})
sns.set_theme(style="ticks", font='serif', font_scale=.8)
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 5], hspace=0.2))
sns.lineplot(data=df_eco.iloc[start_i:end_i], linewidth=1, zorder=0,
             palette=['black', 'purple'],
             ax=ax0)
ax0.set_ylabel('Energy prices\n[EUR/kWh]')
ax0.yaxis.grid(color='0.8', linestyle='dashed', zorder=0)
sns.lineplot(data=df_temps.iloc[start_i:end_i], linewidth=1, zorder=0,
             palette=['green', 'blue', 'grey'],
             ax=ax1)
ax1.set_ylabel('Temperature [$^\circ$C]')
ax1.set_xlabel('Timestamp')
ax1.xaxis.set_tick_params(rotation=90)
ax1.set_xlim(df_temps.index[start_i], df_temps.index[end_i])
ax2 = ax1.twinx()
sns.lineplot(data=df_heat.iloc[start_i:end_i], linewidth=0.85, drawstyle='steps-post', zorder=2,
             linestyle='dashdot',
             palette=['red', 'orange'], alpha=0.3,
             ax=ax2)
ax2.fill_between(df_heat.index[start_i:end_i],
                 [0]*len(df_heat.index[start_i:end_i]),
                 df_heat['$Q_{space heating}$'].iloc[start_i:end_i],
                 step='post',
                 color='red', alpha=0.3)
ax2.fill_between(df_heat.index[start_i:end_i],
                 [0]*len(df_heat.index[start_i:end_i]),
                 df_heat['$Q_{irradiance}$'].iloc[start_i:end_i],
                 step='post',
                 color='orange', alpha=0.3)
ax2.set_ylabel('Heating input [kWh]')
ax1.yaxis.grid(color='0.8', linestyle='dashed', zorder=0)

ax1.set_yticks(np.round(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 5)))
ax2.set_yticks(np.round(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 5), 1))

# added these three lines
lns = ax0.lines + ax1.lines + ax2.lines
labs = [l.get_label() for l in lns]
ax1.legend([], [], frameon=False)
ax2.legend([], [], frameon=False)
ax0.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 2.3),
           ncol=4,)
plt.tight_layout()

# Save
plt.savefig(path_fig+'blg_control_all.pdf', bbox_inches='tight')




########################################################################################################################

# Load curves
path_to_weather = r'C:\.../'
timestamp = pd.date_range(start='01.01.2019', end='01.01.2020', freq='H')
timestamp = timestamp.drop(timestamp[-1])
anonymized_uuids = [x+1 for x in range(len(unique_uuids))]

df_data = dict()
for sa in range(10):
    data_file = path_to_weather + 'scenario_' + str(sa) + '.csv'
    df = pd.read_csv(data_file, index_col=0)
    df.set_index(timestamp, inplace=True)

    df.rename(columns={'Ta': '$T_{ambiant}$ [$^\circ$C]',
                       'Ps': '$Q_{irradiance}$ [kWh]',
                       'Day-ahead Price [EUR/kWh]': '$p_{electricity}$ [EUR/kWh]',
                       'gas_price [EUR/kWh]': '$p_{gas}$ [EUR/kWh]'},
              inplace=True)
    for i, uuid in enumerate(unique_uuids):
        df.rename(columns={uuid+'_T_blg_set': '$T_{set-point}$_'+str(anonymized_uuids[i]),
                           uuid+'_E_elec': '$E_{electricity}$_'+str(anonymized_uuids[i])},
                  inplace=True)

    cols_to_drop = [col for col in df.columns if '_E_elec' in col or '_T_blg_set' in col]
    df.drop(cols_to_drop, axis=1, inplace=True)
    df_data[sa] = df


# Weather
weather_cols = ['$Q_{irradiance}$ [kWh]', '$T_{ambiant}$ [$^\circ$C]']
data = pd.DataFrame()
for sa in range(10):
    data[str(sa+1)] = df_data[sa][weather_cols[0]].sort_values().reset_index(drop=True)

# Irradiance
sns.set(rc={"figure.figsize": (7, 3.5)})
sns.set_theme(style="whitegrid", font='serif', font_scale=.8)
fig, ax0 = plt.subplots()
sns.lineplot(data=data, linewidth=1, palette='rocket').set(title='Global horizontal irradiance [kWh/m$^2$]')
sns.lineplot(data=df_data[sa][weather_cols[0]].reset_index(drop=True), linewidth=1,
             palette='black', alpha=0.2)
ax0.set_xlim(0, len(data))
ax0.legend(title='Scenario', loc='center right', bbox_to_anchor=(1.2, 0.5), ncol=1)
ax0.set_xlabel('Timestep [h]')
plt.tight_layout()

# Ambiant
data = pd.DataFrame()
for sa in range(10):
    data[str(sa+1)] = df_data[sa][weather_cols[1]].sort_values().reset_index(drop=True)
sns.set(rc={"figure.figsize": (7, 3.5)})
sns.set_theme(style="whitegrid", font='serif', font_scale=.8)
fig, ax0 = plt.subplots()
sns.lineplot(data=data, linewidth=1, palette='rocket').set(title='Ambient temperature [$^\circ$C]')
sns.lineplot(data=df_data[sa][weather_cols[1]].reset_index(drop=True), linewidth=1,
             palette='black', alpha=0.2)
ax0.set_xlim(0, len(data))
ax0.legend(title='Scenario', loc='center right', bbox_to_anchor=(1.2, 0.5), ncol=1)
ax0.set_xlabel('Timestep [h]')
plt.tight_layout()



# Multiplot weather
weather_cols = ['$Q_{irradiance}$ [kWh]', '$T_{ambiant}$ [$^\circ$C]']
dataT, dataI = pd.DataFrame(), pd.DataFrame()
for sa in range(10):
    dataI[str(sa+1)] = df_data[sa][weather_cols[0]].sort_values().reset_index(drop=True)
    dataT[str(sa + 1)] = df_data[sa][weather_cols[1]].sort_values().reset_index(drop=True)

sns.set(rc={"figure.figsize": (7, 5)})
sns.set_theme(style="whitegrid", font='serif', font_scale=.8)
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.2))
sns.lineplot(data=dataI, linewidth=1, palette='rocket', alpha=0.5, dashes=False, ax=ax0).set(title='Global horizontal irradiance [kWh/m$^2$]')
sns.lineplot(data=df_data[sa][weather_cols[0]].reset_index(drop=True), linewidth=1,
             palette='black', alpha=0.2, ax=ax0)
sns.lineplot(data=dataT, linewidth=1, palette='rocket', alpha=0.5, dashes=False, ax=ax1).set(title='Ambient temperature [$^\circ$C]')
sns.lineplot(data=df_data[sa][weather_cols[1]].reset_index(drop=True), linewidth=1,
             palette='black', alpha=0.2, ax=ax1)
ax0.set_xlim(0, len(data))
ax0.set_ylabel('')
ax1.set_ylabel('')
ax0.legend([], [], frameon=False)
ax1.legend(title='Scenario', loc='center right', bbox_to_anchor=(1.2, 1.0), ncol=1)
ax1.set_xlabel('Timestep [h]')
plt.tight_layout()

# Save
plt.savefig(path_fig+'load_weather.pdf', bbox_inches='tight')


# Multiplot eco
eco_cols = ['$p_{electricity}$ [EUR/kWh]', '$p_{gas}$ [EUR/kWh]']
dataT, dataI = pd.DataFrame(), pd.DataFrame()
for sa in range(10):
    dataI[str(sa+1)] = df_data[sa][eco_cols[0]].sort_values().reset_index(drop=True)
    dataT[str(sa + 1)] = df_data[sa][eco_cols[1]].sort_values().reset_index(drop=True)

sns.set(rc={"figure.figsize": (7, 5)})
sns.set_theme(style="whitegrid", font='serif', font_scale=.8)
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.2))
sns.lineplot(data=dataI, linewidth=1, palette='rocket', alpha=0.5, dashes=False, ax=ax0).set(title='Electricity day-ahead forecasted price [EUR/kWh]')
sns.lineplot(data=df_data[sa][eco_cols[0]].reset_index(drop=True), linewidth=1,
             palette='black', alpha=0.2, ax=ax0)
sns.lineplot(data=dataT, linewidth=1, palette='rocket', alpha=0.5, dashes=False, ax=ax1).set(title='Gas price [EUR/kWh]')
sns.lineplot(data=df_data[sa][eco_cols[1]].reset_index(drop=True), linewidth=1,
             palette='black', alpha=0.2, ax=ax1)
ax0.set_xlim(0, len(data))
ax0.set_ylabel('')
ax1.set_ylabel('')
ax0.legend([], [], frameon=False)
ax1.legend(title='Scenario', loc='center right', bbox_to_anchor=(1.2, 1.0), ncol=1)
ax1.set_xlabel('Timestep [h]')
plt.tight_layout()

# Save
plt.savefig(path_fig+'load_economic.pdf', bbox_inches='tight')



# Multiplot occ
occ_cols = ['$T_{set-point}$_', '$E_{electricity}$_']
dataT, dataI = pd.DataFrame(), pd.DataFrame()
for uuid in anonymized_uuids:
    dataT_avg, dataI_avg = pd.DataFrame(), pd.DataFrame()
    for sa in range(10):
        dataI_avg[str(sa + 1)] = df_data[sa][occ_cols[0]+str(uuid)].sort_values().reset_index(drop=True)
        dataT_avg[str(sa + 1)] = df_data[sa][occ_cols[1]+str(uuid)].sort_values().reset_index(drop=True)
    dataI[str(uuid)] = dataI_avg.mean(axis=1)
    dataT[str(uuid)] = dataT_avg.mean(axis=1)/1000

sns.set(rc={"figure.figsize": (7, 5)})
sns.set_theme(style="whitegrid", font='serif', font_scale=.8)
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.2))
sns.lineplot(data=dataI, linewidth=1, palette='rocket', alpha=0.5, ax=ax0, dashes=False, zorder=1).set(title='Set-point temperature [$^\circ$C]')
sns.lineplot(data=df_data[sa][occ_cols[0]+str(uuid)].reset_index(drop=True), linewidth=1,
             palette='black', alpha=0.2, ax=ax0)
sns.lineplot(data=dataT, linewidth=1, palette='rocket', alpha=0.5, dashes=False, ax=ax1).set(title='Electric base load [kWh]')
sns.lineplot(data=df_data[sa][occ_cols[1]+str(uuid)].reset_index(drop=True)/1000, linewidth=1,
             palette='black', alpha=0.2, ax=ax1)
ax0.set_xlim(0, len(data))
ax0.set_ylabel('')
ax1.set_ylabel('')
ax0.legend([], [], frameon=False)
ax1.legend(title='Building', loc='center right', bbox_to_anchor=(1.3, 1.1), ncol=2)
ax1.set_xlabel('Timestep [h]')
plt.tight_layout()

# Save
plt.savefig(path_fig+'load_all_occupants.pdf', bbox_inches='tight')
