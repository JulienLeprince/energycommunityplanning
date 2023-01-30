# Parameters

# Economic data
p_co2 = 85.22 / 1000000 * 200.8  # eur/tCo2 * gco2/tco2 * gCO2/kWh
p_slk = 10e3
r = 0.028
A_blg_roof_max = 36  # m2

# Heat pump
C_blg_hp_max = 7
C_blg_hp_min = 0.25
eff_blg_hp = 3.15
hp_lifetime = 16  # years
pi_hp_1, pi_hp_2, pi_hp_3, pi_hp_4 = 13.39, -0.047, 1.109, 0.012
inv_blg_hp = 10950
b_blg_hp = 311   # O&M
inv_lvl_blg_hp = inv_blg_hp*r/(1-(1+r)**(-hp_lifetime))

# Battery
C_blg_bat_max = 6000  # kWh
C_blg_bat_min = 0.1
eff_blg_bat_ch = 0.98
eff_blg_bat_dch = 0.97
decay_blg_bat = 1 - 0.1/24   # %/day/hours = %/hours
power_eff_blg_bat_ch = 0.5
power_eff_blg_bat_dch = 1
inv_blg_bat = 1042  # eur/kWh  inv
a_blg_bat = 0.54  # eur/kW/year O&M
b_blg_bat = 0
bat_lifetime = 20  # years
inv_lvl_blg_bat = inv_blg_bat*r/(1-(1+r)**(-bat_lifetime))

# Thermal energy storage
C_blg_tes_max = 3  # kWh
C_blg_tes_min = 0
eff_blg_tes_ch = 1
eff_blg_tes_dch = 1
decay_blg_tes = 0.021
power_eff_blg_tes_ch = 1
power_eff_blg_tes_dch = 1
tes_lifetime = 30  # years
inv_blg_tes = 410  # EUR/kWh
b_blg_tes = 50   # O&M  eur/unit/year
inv_lvl_blg_tes = inv_blg_tes*r/(1-(1+r)**(-tes_lifetime))

# Gas boiler
C_blg_bol_max = 14
C_blg_bol_min = 0
bol_lifetime = 20  # years
eff_blg_bol = 0.97
inv_blg_bol = 3900/14  # eur/kWh
b_blg_bol = 192   # eur/unit/yr   O&M
inv_lvl_blg_bol = inv_blg_bol*r/(1-(1+r)**(-bol_lifetime))

# Solar thermal collector  - Araz Ashouri Ref. - except price
A_blg_stc_max = 36
A_blg_stc_min = 6
stc_lifetime = 25  # years
eff_blg_stc = 0.6
U_blg_stc = 0.0053  # kW/m2/K
T_stc = 40  # Celsius
inv_blg_stc = 4140/6  # eur/m2
a_blg_stc = 52/6  # eur/m2/yr   O&M
b_blg_stc = 0
inv_lvl_blg_stc = inv_blg_stc*r/(1-(1+r)**(-stc_lifetime))

# Photovoltaics -blg
A_blg_pv_max = 36  # m2
A_blg_pv_min = 4.88  # m2
eff_blg_pv = 0.205  # -
pv_lifetime = 35  # years
inv_blg_pv = (1240+13.4)*6/4.88   # EUR/kW * kW/unit * unit/m2 = EUR/m2  inv
a_blg_pv = 13.4*6/4.88  # eur/kWh/yr * kW/unit * unit/m2 = EUR/m2/yr   O&M
b_blg_pv = 0
inv_lvl_blg_pv = inv_blg_pv*r/(1-(1+r)**(-pv_lifetime))

# Thermal comfort building
T_blg_buffer = 1

# Grid topology
E_hv_max, E_mv_max, E_lv_max = 500000, 400, 102.675   # kW

# Battery - com
C_com_bat_max = 300000  # kWh
C_com_bat_min = 0
C_combat_min = 0.1
eff_com_bat_ch = 0.83
eff_com_bat_dch = 1
decay_com_bat = 0.01   # %/day/hours = %/hours
power_eff_com_bat_ch = 50000/C_com_bat_max
power_eff_com_bat_dch = 50000/C_com_bat_max
inv_com_bat = 370  # eur/kWh  inv
a_com_bat = 1.015  # [% of total investment]  O&M  - to multiply to levelized inv_com_bat
b_com_bat = 0
bat_com_lifetime = 19  # years
inv_lvl_com_bat = inv_com_bat*r/(1-(1+r)**(-bat_com_lifetime))

# Hydrogen seasonal storage
# Electrolyzer
C_com_elec_max = 1000
C_com_elec_min = 0
eff_com_elec_ch = 0.68
power_eff_com_elec_ch = 1
elec_lifetime = 30  # years
inv_com_elec = 570  # eur/kWh
a_com_elec = 28.5  # eur/kw/yr  O&M
b_com_elec = 0
inv_lvl_com_elec = inv_com_elec*r/(1-(1+r)**(-elec_lifetime))

# Fuel Cell
C_com_fc_max = 100  # kWh
C_com_fc_min = 5  # kWh
eff_com_fc_dch = 0.5
power_eff_com_fc_dch = 1
fc_lifetime = 10  # years
inv_com_fc = 1300  # eur/kWh   inv
a_com_fc = 65  # eur/kw/yr  O&M
b_com_fc = 0
inv_lvl_com_fc = inv_com_fc*r/(1-(1+r)**(-fc_lifetime))

# Hydrogen tank
C_com_hyd_max = 16700  # kWh
C_com_hyd_min = 0  # kWh
eff_com_hyd_ch = 0.88*eff_com_elec_ch
eff_com_hyd_dch = 1*eff_com_fc_dch
decay_com_hyd = 1
power_eff_com_hyd_ch = 95/16700
power_eff_com_hyd_dch = 1
hyd_lifetime = 25  # years
inv_com_hyd = 57  # eur/kWh   inv
a_com_hyd = 0.6  # eur/kw/yr  O&M
b_com_hyd = 0
inv_lvl_com_hyd = inv_com_hyd*r/(1-(1+r)**(-hyd_lifetime))

# Photovoltaics - com
A_com_pv_max = 488  # m2
A_com_pv_min = 48.8  # m2
eff_com_pv = 0.205  # -
inv_com_pv = 870*6/4.88   # EUR/kW * kW/unit * unit/m2 = EUR/m2   inv
a_com_pv = 10.6*6/4.88   # EUR/kW * kW/unit * unit/m2 = EUR/m2   O&M
b_com_pv = 0
inv_lvl_com_pv = inv_com_pv*r/(1-(1+r)**(-pv_lifetime))
