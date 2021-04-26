import csv
import math
from termcolor import colored
from utility import validation
from envs.AS_GymEnv import ASGymEnv


def get_axis_of_state(state, x_axis, y_axis):
    obs_location = state

    xy_plane = x_axis * y_axis

    z = math.floor(obs_location / xy_plane)
    x = math.floor((obs_location - (xy_plane * z)) / x_axis)
    y = (obs_location - (xy_plane * z)) - (x * x_axis)

    return [x, y, z]


def read_file(selection=-1):
    print("Reading file...")
    with open("paths.csv") as csv_file:
        path_reader = csv.reader(csv_file, delimiter=',')

        # x_axis, y_axis, z_axis, cube_size, blue_count, red_count, destinations, obstacles, time_taken

        episode = [[0], [0], [0], [0], [0]]
        line = 0
        for row in path_reader:
            if line == 0:
                config = row
                x_axis = int(config[0])
                y_axis = int(config[1])
                z_axis = int(config[2])
                blue_count = int(config[3])
                red_count = int(config[4])
                obstacle_string = config[5]
                time_taken = config[6]
                obstacle_string = obstacle_string.replace('[', '')
                obstacle_string = obstacle_string.replace(']', '')
                obstacle_string = obstacle_string.replace(',', '')
                obstacle_string = obstacle_string.replace("'", '')
                obstacles = obstacle_string.split()
                # states_in_env = int(x_axis * y_axis * z_axis)
                drone_count = blue_count + red_count
            elif line == 1:
                states = row
                for s in range(0, len(states)):
                    states[s] = states[s].replace('[', '')
                    states[s] = states[s].replace(']', '')
                    states[s] = states[s].replace("'", '')
                    states[s] = states[s].split(',')
                    for x in range(0, len(states[s])):
                        states[s][x] = int(states[s][x])
                episode[0] = states
            elif line == 1000:
                states = row
                for s in range(0, len(states)):
                    states[s] = states[s].replace('[', '')
                    states[s] = states[s].replace(']', '')
                    states[s] = states[s].replace("'", '')
                    states[s] = states[s].split(',')
                    for x in range(0, len(states[s])):
                        states[s][x] = int(states[s][x])
                episode[1] = states
            elif line == 100000:
                states = row
                for s in range(0, len(states)):
                    states[s] = states[s].replace('[', '')
                    states[s] = states[s].replace(']', '')
                    states[s] = states[s].replace("'", '')
                    states[s] = states[s].split(',')
                    for x in range(0, len(states[s])):
                        states[s][x] = int(states[s][x])
                episode[2] = states
            elif line == 1000000:
                states = row
                for s in range(0, len(states)):
                    states[s] = states[s].replace('[', '')
                    states[s] = states[s].replace(']', '')
                    states[s] = states[s].replace("'", '')
                    states[s] = states[s].split(',')
                    for x in range(0, len(states[s])):
                        states[s][x] = int(states[s][x])
                episode[3] = states
            elif line == int(selection):
                states = row
                for s in range(0, len(states)):
                    states[s] = states[s].replace('[', '')
                    states[s] = states[s].replace(']', '')
                    states[s] = states[s].replace("'", '')
                    states[s] = states[s].split(',')
                    for x in range(0, len(states[s])):
                        states[s][x] = int(states[s][x])
                episode[4] = states
            line += 1

        return episode, line, time_taken, drone_count, x_axis, y_axis, obstacles, red_count, blue_count


def text_view():
    # read in path file, get important episodes
    episode, line, time_taken, drone_count, x_axis, y_axis, obstacles, red_count, blue_count = read_file()
    selection = validation(("Select a specific episode to view, up to", line), True)
    episode, line, time_taken, drone_count, x_axis, y_axis, obstacles, red_count, blue_count = read_file(selection)

    # viewer main
    # user select episode to view
    quit = False
    while not quit:
        print("Time taken for", line, "episodes: ", time_taken)
        print("Episodes to view...")
        if episode[0] != [0]:
            print("For Episode 1 - Enter 0")
        if episode[1] != [0]:
            print("For Episode 1000 - Enter 1")
        if episode[2] != [0]:
            print("For Episode 100000 - Enter 2")
        if episode[3] != [0]:
            print("For Episode 1000000 - Enter 3")
        if episode[4] != [0]:
            print("For Episode", selection, "- Enter 4")
        user_ep = int(validation("Select an episode to view, or Q to return to menu:", False, "0", "1", "2", "3", "4"))

        # episode viewer
        steps = len(episode[user_ep])
        # steps = int((int(len(episode[int(user_ep)])) - 1) / drone_count)
        print("Steps in episode:", steps)
        wait = input("Press any key to continue through the steps...")
        if (wait == "q") or (wait == "Q"):
            break

        # dest = int(episode[user_ep][0])
        # dest_coords = get_axis_of_state(dest, x_axis, y_axis)
        for step in range(0, steps):
            print("\n\n\n")
            print("Step:", step, " - Number is Z level - Blue = Team1, Red = Team2, Green = Target")

            dest = episode[user_ep][step][0]
            dest_coords = get_axis_of_state(dest, x_axis, y_axis)

            drones = []
            coords = []
            for d in range(0, drone_count):
                drones.append(episode[user_ep][step][d+1])
                coords.append(get_axis_of_state(drones[d], x_axis, y_axis))

            if step == 0:
                obstacle_coords = []
                for i in range(0, len(obstacles)):
                    obstacle_coords.append(get_axis_of_state(int(obstacles[i]), x_axis, y_axis))

            for x in range(1, x_axis + 1):
                for y in range(0, y_axis):
                    overlap = False
                    if int(dest_coords[0]) == (x_axis - x) and int(dest_coords[1]) == y:
                        print(" ", end='')
                        print(colored(int(dest_coords[2]), 'green'), end='')
                        print(" ", end='')
                        overlap = True
                    if (blue_count >= 1) and not overlap:
                        if coords[0][0] == (x_axis - x) and coords[0][1] == y:
                            print(" ", end='')
                            print(colored(coords[0][2], 'blue'), end='')
                            print(" ", end='')
                            overlap = True
                    if (blue_count >= 2) and not overlap:
                        if coords[1][0] == (x_axis - x) and coords[1][1] == y:
                            print(" ", end='')
                            print(colored(coords[1][2], 'blue'), end='')
                            print(" ", end='')
                            overlap = True
                    if (red_count >= 1) and not overlap:
                        if blue_count == 1:
                            drone_i = 1
                        elif blue_count == 2:
                            drone_i = 2
                        if coords[drone_i][0] == (x_axis - x) and coords[drone_i][1] == y:
                            print(" ", end='')
                            print(colored(coords[drone_i][2], 'red'), end='')
                            print(" ", end='')
                            overlap = True
                    if (red_count >= 2) and not overlap:
                        if blue_count == 1:
                            drone_i = 2
                        elif blue_count == 2:
                            drone_i = 3
                        if coords[drone_i][0] == (x_axis - x) and coords[drone_i][1] == y:
                            print(" ", end='')
                            print(colored(coords[drone_i][2], 'red'), end='')
                            print(" ", end='')
                            overlap = True
                    if not overlap:
                        if not obstacle_coords == []:
                            for i in range(0, len(obstacle_coords)):
                                if int(obstacle_coords[i][0]) == (x_axis - x) and int(obstacle_coords[i][1]) == y:
                                    print(" ", end='')
                                    print(colored(int(obstacle_coords[i][2]), 'yellow'), end='')
                                    print(" ", end='')
                                    overlap = True
                        if not overlap:
                            print(" - ", end='')
                print("")

            wait = input("Press any key to continue through the steps...")
            if wait == "q" or wait == "Q":
                break


def sim_view():
    # read in path file, get important episodes
    episode, line, time_taken, drone_count, x_axis, y_axis, obstacles, red_count, blue_count = read_file()
    selection = validation(("Select a specific episode to view, up to", line), True)
    episode, line, time_taken, drone_count, x_axis, y_axis, obstacles, red_count, blue_count = read_file(selection)

    # viewer main
    # user select episode to view
    quit = False
    while not quit:
        print("Time taken for", line, "episodes: ", time_taken)
        print("Episodes to view...")
        if episode[0] != [0]:
            print("For Episode 1 - Enter 0")
        if episode[1] != [0]:
            print("For Episode 1000 - Enter 1")
        if episode[2] != [0]:
            print("For Episode 100000 - Enter 2")
        if episode[3] != [0]:
            print("For Episode 1000000 - Enter 3")
        if episode[4] != [0]:
            print("For Episode", selection, "- Enter 4")
        user_ep = int(validation("Select an episode to view, or Q to return to menu:", False, "0", "1", "2", "3", "4"))

        # episode viewer
        steps = len(episode[user_ep])
        print("Steps in episode:", steps)
        wait = input("WARNING: Ensure your AirSim Unreal Environment is running")
        if (wait == "q") or (wait == "Q"):
            break

        # connect to airsim
        env = ASGymEnv(blue_count, red_count, 0, False)

        for step in range(0, steps):
            for d in range(0, drone_count + 1):
                state = episode[user_ep][step][d]
                pos = env.get_coords_of_state(state)
                if d == 0:
                    env.drone_dest.move(pos)
                elif d == 1:
                    env.drone_agent1.move(pos)
                elif d == 2:
                    env.drone_agent2.move(pos)
                elif d == 3:
                    env.drone_agent3.move(pos)
                elif d == 4:
                    env.drone_agent4.move(pos)

            wait = input("Press any key to continue through the steps...")
            if wait == "q" or wait == "Q":
                break
