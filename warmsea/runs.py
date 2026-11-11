"""Run names used in the paper, the dataset and the scripts."""

# SST experiments (Tables S2 and S3). Name: [physics_]sst_member; physics is Morrison-YSU unless it starts with tmynn.
SST_RUNS = {}
for sst, s in (('clim', 'sstclim'), ('curr', 'ctrl'), ('plus2k', 'plus2')):
    SST_RUNS[f'{sst}_ctrl'] = f'init12_{s}'
    SST_RUNS[f'{sst}_skb1'] = f'skebs1_{s}'
    SST_RUNS[f'{sst}_skb2'] = f'skebs2_{s}'
    SST_RUNS[f'tmynn_{sst}_ctrl'] = f'tmynn_{s}'
    SST_RUNS[f'tmynn_{sst}_skb1'] = f'tmynn_skebs1_{s}'
    SST_RUNS[f'tmynn_{sst}_skb2'] = f'tmynn_skebs2_{s}'
SST_RUNS.update({'plus2kall_ctrl': 'warm2_ctrl', 'plus2kall_skb1': 'warm2_skebs1', 'plus2kall_skb2': 'warm2_skebs2'})

# Configuration selection (Table S1): forcing, microphysics and boundary layer. The two ERA5T rows not listed here are
# runs of the SST experiments: ERA5T Morrison-YSU is curr_ctrl and ERA5T Thompson-MYNN is tmynn_curr_ctrl.
SELECTION_RUNS = {
    f'sel_{f}_{mp}_{pbl}': f'{f}_{mp[0]}{pbl}'
    for f in ('gfs', 'ifs')
    for mp in ('thompson', 'morrison')
    for pbl in ('mynn', 'ysu')
}
SELECTION_RUNS.update({'sel_era5t_morrison_mynn': 'era5t_mmynn', 'sel_era5t_thompson_ysu': 'era5t_tysu'})

# Map panels (Figures 4 and S4) and sections (Figure 4g,h): run and time (UTC).
MAP_TIMES = {
    'curr_skb1': '2026-10-04 03:00',
    'plus2k_skb1': '2026-10-04 17:00',
    'plus2k_ctrl': '2026-10-04 06:00',
    'plus2k_skb2': '2026-10-03 22:00',
}
SECTIONS = {'curr_skb1': '2026-10-04 03:00', 'plus2k_skb1': '2026-10-04 17:00'}
