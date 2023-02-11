import numpy as np
import scipy.signal as signal
import pandas as pd
import data_in_out as IO

def calc_player_velocities(team_tracking_data, smoothing=True, window=7, maxspeed=12):
    team_tracking_data = remove_player_velocities(team_tracking_data)
    players_ids = np.unique([ c[:-2] for c in team_tracking_data.columns if c[:6] in ['Team_A', 'Team_B'] and "speed" not in c[7:-2] ])
    dt = team_tracking_data['Time [s]'].diff()
    # second_half_idx = team_tracking_data.Period.idxmax(2)
    for player in players_ids:
        vx = team_tracking_data[player + '_x'].diff() / dt
        vy = team_tracking_data[player + '_y'].diff() / dt

        # speed > maxspeed is likely error recording
        # if maxspeed > 0:
        #     raw_speed = team_tracking_data[player + "_speedKmh"]
        #     vx.values[raw_speed > maxspeed] = np.nan
        #     vy.values[raw_speed > maxspeed] = np.nan
        
        # TODO: Smoothing
        team_tracking_data[player + "_vx"] = pd.to_numeric(vx)
        team_tracking_data[player + "_vy"] = pd.to_numeric(vy)
    return team_tracking_data

def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    team = team.drop(columns=columns)
    return team