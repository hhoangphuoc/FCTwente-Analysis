import numpy as np
import pitchcontrol as pc
import data_in_out as io

def load_xT_grid(fname='xT_grid.csv'):
    """ load_xT_grid(fname='xT.csv')
    
    # load pregenerated xT surface from file. 
    
    Parameters
    -----------
        fname: filename & path of xT grid (default is 'xT_grid.csv' in the curernt directory)
        
    Returns
    -----------
        xT: The xT surface (default is a (32,50) grid)
    
    """
    xT = np.loadtxt(fname, delimiter=',')
    return xT
    
def get_xT_at_location(position,xT,attack_direction,field_dimen=(105.,68.)):
    """ get_xT_at_location
    
    Returns the xT value at a given (x,y) location
    
    Parameters
    -----------
        position: Tuple containing the (x,y) pitch position
        xT: tuple expected threat value grid (loaded using load_xT_grid() )
        attack_direction: Sets the attack direction (1: left->right, -1: right->left)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (105,68)
            
    Returrns
    -----------
        xT value at input position
        
    """
    
    x,y = position
    if abs(x)>field_dimen[0]/2. or abs(y)>field_dimen[1]/2.:
        return 0.0 # Position is off the field, xT is zero
    else:
        if attack_direction==-1:
            xT = np.fliplr(xT)
        ny,nx = xT.shape
        dx = field_dimen[0]/float(nx)
        dy = field_dimen[1]/float(ny)
        ix = (x+field_dimen[0]/2.-0.0001)/dx
        iy = (y+field_dimen[1]/2.-0.0001)/dy
        return xT[int(iy),int(ix)]
    
def calculate_action_value_added(event_id, events, tracking_home, tracking_away, GK_numbers, xT, params):
    """ calculate_xT_added
    
    Calculates the expected threat value added by a pass
    
    Parameters
    -----------
        event_id: Index (not row) of the pass event to calculate xT-added score
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        xT: tuple expected threat value grid (loaded using load_xT_grid() )
        params: Dictionary of pitch control model parameters (default model parameters can be generated using default_model_params() )
        
    Returrns
    -----------
        action_value_added: Expected xT value-added of pass defined by event_id
        xT_difference: The raw change in xT (ignoring pitch control) between end and start points of pass
    """
    # pull out pass details from the event data
    pass_start_pos = np.array([events.loc[event_id]['start_x'],events.loc[event_id]['start_y']])
    pass_target_pos = np.array([events.loc[event_id]['end_x'],events.loc[event_id]['end_y']])
    pass_frame = events.loc[event_id]['start_frameID'] 
    pass_team = events.loc[event_id].Team
    
    # direction of play for atacking team (so we know whether to flip the xT grid)
    home_attack_direction = io.find_playing_direction(tracking_home,'Team_A')
    if pass_team=='Team_A':
        attack_direction = home_attack_direction
        attacking_players = pc.initialise_players(tracking_home.loc[pass_frame],'Team_A',params,GK_numbers[0])
        defending_players = pc.initialise_players(tracking_away.loc[pass_frame],'Team_B',params,GK_numbers[1])
    elif pass_team=='Team_B':
        attack_direction = home_attack_direction*-1
        defending_players = pc.initialise_players(tracking_home.loc[pass_frame],'Team_A',params,GK_numbers[0])
        attacking_players = pc.initialise_players(tracking_away.loc[pass_frame],'Team_B',params,GK_numbers[1])    
    # flag any players that are offside
    attacking_players = pc.check_offsides( attacking_players, defending_players, pass_start_pos, GK_numbers)
    # pitch control grid at pass start location
    pitchcontrol_start,_ = pc.calculate_pitch_control_at_target(pass_start_pos, attacking_players, defending_players, pass_start_pos, params)

    # pitch control grid at pass end location
    pitchcontrol_target,_ = pc.calculate_pitch_control_at_target(pass_target_pos, attacking_players, defending_players, pass_start_pos, params)

    # xT at start location
    xT_start = get_xT_at_location(pass_start_pos, xT, attack_direction=attack_direction)

    # xT at end location
    xT_target   = get_xT_at_location(pass_target_pos,xT,attack_direction=attack_direction)

    # 'Expected' xT at target and start location
    action_value_target = pitchcontrol_target*xT_target

    action_value_start = pitchcontrol_start*xT_start

    # difference is the (expected) xT added
    action_value_added = action_value_target - action_value_start

    # print("Pitch control start = " + str(pitchcontrol_start))
    # print("Pitch control target = " + str(pitchcontrol_target))
    # print("xT_start = " + str(xT_start))
    # print("xT_target = " + str(xT_target))
    # print("action_value_start = " + str(action_value_start))
    # print("action_value_target = " + str(action_value_target))
    
    
    # Also calculate the straight up change in xT
    xT_difference = xT_target - xT_start

    return action_value_added, xT_difference

def find_max_value_added_target( event_id, events, tracking_home, tracking_away, GK_numbers, xT, params ):
    """ find_max_value_added_target
    
    Finds the *maximum* expected threat value that could have been achieved for a pass (defined by the event_id) by searching the entire field for the best target.
    
    Parameters
    -----------
        event_id: Index (not row) of the pass event to calculate xT-added score
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        xT: tuple expected threat value grid (loaded using load_xT_grid() )
        params: Dictionary of pitch control model parameters (default model parameters can be generated using default_model_params() )
        
    Returrns
    -----------
        maxxT_added: maximum xT value-added that could be achieved at the current instant
        max_target_location: (x,y) location of the position of the maxxT_added
    """
    # pull out pass details from the event data
    pass_start_pos = np.array([events.loc[event_id]['start_x'],events.loc[event_id]['start_y']])
    pass_frame = events.loc[event_id]['start_frameID']
    pass_team = events.loc[event_id].Team
    
    # direction of play for atacking team (so we know whether to flip the xT grid)
    home_attack_direction = io.find_playing_direction(tracking_home,'Team_A')
    if pass_team=='Team_A':
        attack_direction = home_attack_direction
        attacking_players = pc.initialise_players(tracking_home.loc[pass_frame],'Team_A',params,GK_numbers[0])
        defending_players = pc.initialise_players(tracking_away.loc[pass_frame],'Team_B',params,GK_numbers[1])
    elif pass_team=='Team_B':
        attack_direction = home_attack_direction*-1
        defending_players = pc.initialise_players(tracking_home.loc[pass_frame],'Team_A',params,GK_numbers[0])
        attacking_players = pc.initialise_players(tracking_away.loc[pass_frame],'Team_B',params,GK_numbers[1])   
        
    # flag any players that are offside
    attacking_players = pc.check_offsides( attacking_players, defending_players, pass_start_pos, GK_numbers)
    
    # pitch control grid at pass start location
    pitchcontrol_start,_ = pc.calculate_pitch_control_at_target(pass_start_pos, attacking_players, defending_players, pass_start_pos, params)
    
    # xT at start location
    xT_start = get_xT_at_location(pass_start_pos, xT, attack_direction=attack_direction)

    # calculate pitch control surface at moment of the pass
    PPCF,xgrid,ygrid = pc.generate_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, GK_numbers, field_dimen = (105.,68.,), n_grid_cells_x = 50, offsides=True)
    
    # xT surface at instance of the pass
    if attack_direction == -1:
        action_value = np.fliplr(xT)*PPCF
    else:
        action_value = xT*PPCF
        
    # find indices of the maxxT
    maxxT_idx = np.unravel_index(action_value.argmax(),action_value.shape)
    
    # Expected xT at current ball position   
    action_value_start = pitchcontrol_start*xT_start
    
    # maxxT_added (difference between max location and current ball location)
    maxxT_added = action_value.max() - action_value_start
    
    # location of maximum
    max_target_location = (xgrid[maxxT_idx[1]], ygrid[maxxT_idx[0]])

    return maxxT_added, max_target_location