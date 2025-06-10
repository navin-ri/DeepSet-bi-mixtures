from chemicals.critical import critical_data_Yaws
df = critical_data_Yaws
df.to_csv('pure_crit_param.csv', index = False)