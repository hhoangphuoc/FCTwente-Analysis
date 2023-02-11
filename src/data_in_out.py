import csv
import os
import numpy as np
import pandas as pd


from dotenv import load_dotenv

load_dotenv()


FC_TWENTE_FOLDER = os.path.join(os.getcwd(), "FC_TWENTE_FOLDER")


def processing_events_data(events, game_id):

    fc_twente_folder = FC_TWENTE_FOLDER

    start_frames = []

    for index, row in events.iterrows():
        # print(index)
        minutes = row['minute']
        seconds = row['second']
        start_x = row['start_x']
        start_y = row['start_y']
        # end_x = row['end_x']
        # end_y = row['end_y']
        player_name = row['FullName']
        team_name = row['Team']
        csv_path = os.path.join(
            fc_twente_folder, f"Game {game_id}", "Players", f"{player_name}.csv",)
        player_data = pd.read_csv(csv_path)

        csv_path = os.path.join(
            fc_twente_folder, f"Game {game_id}", "Players", "Ball.csv",)
        ball_data = pd.read_csv(csv_path)

        start_frame = None
        filter_player_frames = player_data.loc[(
            player_data['Minutes'] == minutes) & (player_data['Seconds'] == seconds)]

        filter_ball_frames = ball_data.loc[(ball_data['Minutes'] == minutes) & (
            ball_data['Seconds'] == seconds)]
        start_frame = filter_ball_frames.iloc[0]['frameID']
        dframe = pd.DataFrame()
        dframe['player_x'] = pd.to_numeric(filter_player_frames["X"])
        dframe['player_y'] = pd.to_numeric(filter_player_frames["Y"])
        dframe['ball_x'] = pd.to_numeric(filter_ball_frames["X"])
        dframe['ball_y'] = pd.to_numeric(filter_ball_frames["Y"])
        dframe['team'] = filter_ball_frames["Team with the ball"]
        dframe['frameID'] = pd.to_numeric(
            filter_ball_frames["frameID"], downcast="integer")
        minDiff = 1000
        final_frame = start_frame
        for index, row in dframe.iterrows():
            if ((abs(row['ball_x'] - row['player_x']) + abs(row['ball_y'] - row['player_y'])+abs(row['ball_x'] - start_x) + abs(row['ball_y'] - start_y) + abs(row['player_x'] - start_x) + abs(row['player_y'] - start_y)) < minDiff) and (row['team'] == team_name):
                minDiff = abs(row['ball_x'] - row['player_x']) + abs(row['ball_y'] - row['player_y'])+abs(row['ball_x'] -
                                                                                                          start_x) + abs(row['ball_y'] - start_y) + abs(row['player_x'] - start_x) + abs(row['player_y'] - start_y)
                final_frame = row["frameID"]
        start_frames.append(final_frame)
    start_frames = pd.DataFrame(start_frames, columns=['start_frameID'])
    events['start_frameID'] = pd.to_numeric(
        start_frames["start_frameID"], downcast="integer")
    return events


def find_playing_direction(team, teamname):
    '''
    Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
    '''
    GK_column_x = teamname+"_"+find_goalkeeper(team)+"_x"
    # +ve is left->right, -ve is right->left
    return -np.sign(team.iloc[0][GK_column_x])


def find_goalkeeper(team):
    '''
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    '''
    x_columns = [c for c in team.columns if c[-2:].lower() ==
                 '_x' and c[:6] in ['Team_A', 'Team_B']]
    GK_col = team.iloc[0][x_columns].abs().idxmax()
    return str(GK_col.split('_')[2] + "_" + GK_col.split('_')[3])


def load_fc_twente_data(
    fc_twente_folder=FC_TWENTE_FOLDER,
    game_id=1,
    team_id=None,
    player_id=None,
    mode="load-event",
):
    if game_id is not None:

        print(f"Loading Game {game_id} data...")

        if mode == "load-process-event":
            print("Loading process event data")
            csv_path = os.path.join(
                os.getcwd(), "processed_events.csv")
            data = pd.read_csv(csv_path)
            return data
        
        if mode == "load-event":
            print(f"Loading event data of Game {game_id}...")
            csv_path = os.path.join(
                fc_twente_folder, f"Game {game_id}", "events.csv")
            data = pd.read_csv(csv_path)
            # processed_data = processing_events_data(data, game_id)
            # return processed_data
            return data

        elif mode == "load-metadata":
            print(f"Loading metadata of Game {game_id}...")
            csv_path = os.path.join(
                fc_twente_folder, f"Game {game_id}", "Metadata.csv")
            return pd.read_csv(csv_path)
        elif mode == "load-ball-data":
            print(f"Loading ball data of Game {game_id}...")
            csv_path = os.path.join(
                fc_twente_folder, f"Game {game_id}", "Players", "Ball.csv"
            )
            return pd.read_csv(csv_path)

        elif mode == "load-team-data":
            print(f"Loading team {team_id} data of Game {game_id}...")
            # print(fc_twente_folder)
            dir = os.path.join(fc_twente_folder, f"Game {game_id}", "Players")
            # print(dir)
            
            list_frames = []
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith(".csv"):
                        if "Ball" not in str(file) and str(team_id) in str(file) and "checkpoint" not in str(file):
                            dframe = pd.DataFrame()
                            params = str(file).split("_")
                            player_id_has_csv = params[3]
                            player_id = player_id_has_csv.split(".")[0]
                            # print(player_id)
                            csv_path = os.path.join(
                                fc_twente_folder,
                                f"Game {game_id}",
                                "Players",
                                f"Team_{team_id}_Player_{player_id}.csv",
                            )
                            data = pd.read_csv(csv_path)
                            # playername =
                            param_x = f"Team_{team_id}_Player_{player_id}_x"
                            param_y = f"Team_{team_id}_Player_{player_id}_y"
                            param_speed = f"Team_{team_id}_Player_{player_id}_speedMs"
                            dframe[param_x] = pd.to_numeric(data["X"])
                            dframe[param_y] = pd.to_numeric(data["Y"])
                            dframe[param_speed] = pd.to_numeric(
                                data["Snelheid"])
                            dframe['frameID'] = pd.to_numeric(
                                data["frameID"], downcast="integer")
                            dframe.set_index('frameID', inplace = True)
                            list_frames.append(dframe)
                            
                        elif "Ball" in str(file):
                            csv_path = os.path.join(
                                fc_twente_folder,
                                f"Game {game_id}",
                                "Players",
                                "Ball.csv",
                            )
                            data = pd.read_csv(csv_path)
                            dframe = pd.DataFrame()
                            dframe["Period"] = pd.to_numeric(data["Period"])
                            dframe["Time [s]"] = pd.to_numeric(
                                data["Time [s]"])
                            dframe["Ball_x"] = pd.to_numeric(data["X"])
                            dframe["Ball_y"] = pd.to_numeric(data["Y"])
                            dframe['frameID'] = pd.to_numeric(
                                data["frameID"], downcast="integer")
                            dframe.set_index('frameID', inplace = True)
                            list_frames.append(dframe)
            result = pd.concat(list_frames, axis=1)
            return result

        # TODO: Loading team data - read multiple csv files
        # elif mode == "load-team-data":
        #     print(f"Loading Team {team_id} data of Game {game_id}...")
        #     csv_path = os.path.join(
        #         fc_twente_folder, f"Game {game_id}", "Players", "Team.csv")
        #     return pd.read_csv(csv_path)

        elif mode == "load-player-data":
            if player_id is not None and team_id is not None:

                print(f"Loading Team {team_id}: player{player_id} data")
                csv_path = os.path.join(
                    fc_twente_folder,
                    f"Game {game_id}",
                    "Players",
                    f"Team_{team_id}_Player_{player_id}.csv",
                )
                return pd.read_csv(csv_path)
            else:
                print("Please provide team_id and player_id")
    else:
        print("Unable to load data if the game_id is not given")
        return None

    return pd.read_csv(csv_path)


def to_metric_coordinates(data, field_dimen=(105.0, 68.0)):
    x_columns = [c for c in data.columns if c[-1].lower() == "x"]
    y_columns = [c for c in data.columns if c[-1].lower() == "y"]
    data[x_columns] = (data[x_columns]) - 0.5 * field_dimen[0]
    data[y_columns] = (data[y_columns]) - 0.5 * field_dimen[1]

    return data
