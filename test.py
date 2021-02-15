observation_space = 300
destinations = [1, 2, 3, 4]
observation = 76
action = 5
x_axis = 5
y_axis = 5
z_axis = 3

states_in_env = observation_space / len(destinations)
states_by_dests = states_in_env

i = 1
while observation > states_by_dests:
    i += 1
    states_by_dests = states_by_dests * i

obs_location = observation - (states_by_dests - states_in_env)

xy_plane = x_axis * y_axis

if action == 1:
    # is drone at max x
    if obs_location <= x_axis:
        print("cant move 1")
    for z in range(1, z_axis - 1):
        if (xy_plane * z) < obs_location <= ((xy_plane * z) + x_axis):
            print("cant move 1")
elif action == 2:
    # is drone at min x
    if (xy_plane - x_axis) < obs_location <= xy_plane:
        print("cant move 2")
    for z in range(1, z_axis - 1):
        if ((xy_plane - x_axis) * z) < obs_location <= (xy_plane * z):
            print("cant move 2")
elif action == 3:
    # is drone at max y
    if obs_location % x_axis == 0:
        print("cant move 3")
elif action == 4:
    # is drone at min y
    if (obs_location - 1) % x_axis == 0:
        print("cant move 4")
elif action == 5:
    # is drone at max z
    if obs_location <= xy_plane:
        print("cant move 5")
elif action == 6:
    # is drone at min z
    if (xy_plane * (z_axis - 1)) < obs_location <= (xy_plane * z_axis):
        print("cant move 6")