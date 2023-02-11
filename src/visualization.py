import data_in_out as IO
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import matplotlib.animation as animation


def plot_frame(
    hometeam,
    awayteam,
    ball,
    figax=None,
    team_colors=("r", "b"),
    field_dimen=(105.0, 68.0),
    include_player_velocities=False,
    annotate = True
):
    if figax is None:  # create new pitch
        fig, ax = plot_pitch(field_dimen=field_dimen)
    else:  # overlay on a previously generated pitch
        fig, ax = figax  # unpack tuple
    for team, color in zip([hometeam, awayteam], team_colors):
        # print(team)
        x_columns = [c for c in team.keys() if c[-2:].lower() == "_x" and c != "Ball_x"]
        y_columns = [c for c in team.keys() if c[-2:].lower() == "_y" and c != "Ball_y"]
        ax.plot(team[x_columns], team[y_columns], color + "o", alpha=0.7)
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            # print(vx_columns)
            ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=0.7)
        if annotate:
            [ ax.text( team[x]+0.5, team[y]+0.5, x.split('_')[3], fontsize=10, color=color  ) for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ]
    ax.plot( ball['X'], ball['Y'], 'ko', alpha=1.0)
    return fig, ax


def plot_pitch(
    field_dimen=(105.0, 68.0),
    field_color="green",
    linewidth=2,
    markersize=20,
):
    fig, ax = plt.subplots(figsize=(12, 8))  # create a figure
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color == "green":
        ax.set_facecolor("mediumseagreen")
        lc = "whitesmoke"  # line color
        pc = "w"  # 'spot' colors
    elif field_color == "white":
        lc = "k"
        pc = "k"
    # ALL DIMENSIONS IN m
    border_dimen = (3, 3)  # include a border arround of the field of width 3m
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    half_pitch_length = field_dimen[0] / 2.0  # length of half pitch
    half_pitch_width = field_dimen[1] / 2.0  # width of half pitch
    signs = [-1, 1]
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    D_length = 8 * meters_per_yard
    D_radius = 10 * meters_per_yard
    D_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard
    # plot half way line # center circle
    ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth)
    ax.scatter(0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize)
    y = np.linspace(-1, 1, 50) * centre_circle_radius
    x = np.sqrt(centre_circle_radius**2 - y**2)
    ax.plot(x, y, lc, linewidth=linewidth)
    ax.plot(-x, y, lc, linewidth=linewidth)
    for s in signs:  # plots each line seperately
        # plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            lc,
            linewidth=linewidth,
        )
        # goal posts & line
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-goal_line_width / 2.0, goal_line_width / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth,
        )
        # 6 yard box
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [-box_width / 2.0, -box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * box_length,
                s * half_pitch_length - s * box_length,
            ],
            [-box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        # penalty area
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [-area_width / 2.0, -area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * area_length,
                s * half_pitch_length - s * area_length,
            ],
            [-area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        # penalty spot
        ax.scatter(
            s * half_pitch_length - s * penalty_spot,
            0.0,
            marker="o",
            facecolor=lc,
            linewidth=0,
            s=markersize,
        )
        # corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(
            s * half_pitch_length - s * x,
            -half_pitch_width + y,
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            s * half_pitch_length - s * x, half_pitch_width - y, lc, linewidth=linewidth
        )
        # draw the D
        y = (
            np.linspace(-1, 1, 50) * D_length
        )  # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2 - y**2) + D_pos
        ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2.0 + border_dimen[0]
    ymax = field_dimen[1] / 2.0 + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)
    return fig, ax

def plot_events(events, figax=None, field_dimen = (105.0,68), indicators = ['Marker','Arrow'], color='r', marker_style = 'o', alpha = 0.5, annotate=False):
    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax 
    for i,row in events.iterrows():
        if 'Marker' in indicators:
            ax.plot(  row['start_x'], row['start_y'], color+marker_style, alpha=alpha )
        if 'Arrow' in indicators:
            ax.annotate("", xy=row[['end_x','end_y']], xytext=row[['start_x','start_y']], alpha=alpha, arrowprops=dict(alpha=alpha,width=0.5,headlength=4.0,headwidth=4.0,color=color),annotation_clip=False)
        if annotate:
            textstring = row['type_name'] + ': ' + row['FullName']
            ax.text( row['start_x'], row['start_y'], textstring, fontsize=10, color=color)
    return fig,ax

def plot_pitchcontrol_for_event(eid, events, home_data, away_data, ball, PPCF, alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (105.0,68)):
    start_frame = events.loc[eid]['start_frameID']
    team_performed = events.loc[eid].Team

    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)

    plot_frame(home_data.loc[start_frame], away_data.loc[start_frame], ball.loc[start_frame], figax=(fig, ax), include_player_velocities=include_player_velocities, annotate=annotate)
    plot_events( events.loc[eid:eid], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )

    if team_performed=='Home':
        cmap = 'bwr_r'
    else:
        cmap = 'bwr'
    
    ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)

    return fig, ax


def generate_video(home, away, ball, path="", file_name="video_out", fps=25, figax=None, team_colors=("r", "b"), field_dimen=(106.0, 68.0), include_player_velocities=False, annotate = True,):
    assert np.all( home.index==away.index)
    index = home.index
    FFMPEGWriter = animation.writers['ffmpeg']
    writer = FFMPEGWriter(fps=fps)
    fname = path + '/' +  file_name + '.mp4'
    if figax is None:
        fig,ax = vis.plot_pitch(field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = []
            for team,color in zip( [home.loc[i],away.loc[i]], team_colors) :
                x_columns = [c for c in team.keys() if c[-2:].lower() == "_x" and c != "Ball_x"]
                y_columns = [c for c in team.keys() if c[-2:].lower() == "_y" and c != "Ball_y"]
                objs, = ax.plot(team[x_columns], team[y_columns], color + "o", alpha=0.7)
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                    # print(vx_columns)
                    objs = ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=0.7)
                    figobjs.append(objs)
            objs,  = ax.plot( ball['X'].loc[i], ball['Y'].loc[i], 'ko', alpha=1.0)
            figobjs.append(objs)

            frame_minute =  int( team['Time [s]']/60. )
            frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
            timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
            objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
            figobjs.append(objs)
            writer.grab_frame()

            for figobj in figobjs:
                figobj.remove()
    print("done")
    # plt.clf()
    # plt.close(fig)    