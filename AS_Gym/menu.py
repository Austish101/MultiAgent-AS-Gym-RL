import training
import viewer
from utility import validation


# main menu
custom_set = False
quit = False
while not quit:
    print("                     -- Main Menu --                     ")
    print("Please select an option below by entering the number:")
    print("1. Train using default settings in AirSim")
    print("2. Train using default settings without simulation")
    print("3. Train using custom settings")
    print("4. Evaluation Mode")
    print("5. View the results of the last training session in text UI")
    print("6. View the results of the last training session in AirSim")
    print("7. Change the environment settings")
    print("Enter Q at any time to quit")
    menu_in = input()

    if menu_in == "1":
        # train in airsim
        wait = input("WARNING: Ensure your AirSim Unreal Environment is running")
        training.training_loop(True)

    elif menu_in == "2":
        # train without sim
        training.training_loop(False)

    elif menu_in == "3":
        # custom settings
        # if user has run custom settings, offer rerun
        if custom_set:
            if validation("Rerun with the previous custom settings? Y/N", False, "Y", "N") == 0:
                # run with last settings
                training.training_loop(is_airsim_in, reuse_in, blue_in, red_in, episodes_in, steps_in, paths_in)
                continue
            else:
                print("Continuing to setting selection...")

        custom_set = True
        # learning on?
        if validation("Agents learning? Y/N", False, "Y", "N") == 0:
            is_learning_in = True
        else:
            is_learning_in = False
        # airsim/not
        if validation("Simulated in AirSim? Y/N", False, "Y", "N") == 0:
            is_airsim_in = True
        else:
            is_airsim_in = False
        # reuse q-tables if possible/applicable
        if validation("Reuse Q-Tables from previous training run? Y/N", False, "Y", "N") == 0:
            reuse_in = True
        else:
            reuse_in = False
        # number of agents
        # blue = target finding
        blue_in = validation("Number of target finding agents? 1/2", False, "1", "2") + 1
        # red = counter
        red_in = validation("Number of counter agents? 0/1/2", False, "0", "1", "2")
        # episodes
        episodes_in = int(validation("Number of episodes to train?", True))
        # steps before timeout
        steps_in = int(validation("Number of steps in episode?", True))
        # save paths?
        if validation("Save the paths taking during training? Will allow viewing in post. Y/N", False, "Y", "N") == 0:
            paths_in = True
        else:
            paths_in = False
        # obstacles if noSim? Obstacles using AirSim must be setup in Unreal
        obs_rate_in = 0
        if not is_airsim_in:
            obs_percent_in = int(validation("Percentage states of random obstacles?", True))
            if obs_percent_in != 0:
                obs_rate_in = float(obs_percent_in / 100)
        # which destination - moving?
        if validation("Moving destination flag? Y/N", False, "Y", "N") == 0:
            is_moving_in = True
        else:
            is_moving_in = False
        # type of agent (advanced, q-learning)
        type_in = validation("Type of learning agent? \n"
                             "0 = Q-learning \n"
                             "1 = Q-Learning vs moving Obstacles \n",
                             False, "0", "1")
        # extra observations
            # TODO

        if is_airsim_in:
            print("WARNING: Ensure your AirSim Unreal Environment is running")
        input("Press any key to begin training...")
        training.training_loop(is_airsim_in, reuse_in, blue_in, red_in, episodes_in, steps_in, paths_in, obs_rate_in,
                               type_in, is_learning_in, is_moving_in)

    elif menu_in == "4":
        if custom_set:
            if validation("Validate using previous settings, reusing Agents? Y/N", False, "Y", "N") == 0:
                # run with last settings
                training.training_loop(is_airsim_in, True, blue_in, red_in, episodes_in, steps_in, paths_in,
                                       obs_rate_in, type_in, False)
                continue

        # preset settings for testing
        print("Preset selection:")
        print("1. Basic Q-Learning, no sim, 2 targeting agents, environment obstacles, 100,000 episodes")
        print("2. Basic Q-Learning, no sim, 2 targeting agents, 1 moving obstacle, 100,000 episodes")
        print("3. Basic Q-Learning, no sim, 2 targeting agents, 1 learning drone, 100,000 episodes")
        print("4. Basic Q-Learning, no sim, 2 targeting agents, 2 learning drones, 100,000 episodes")
        print("5. Basic Q-Learning, no sim, 2 targeting agents, 2 learning drones, environment obstacles, 100,000 episodes")
        preset_in = int(validation("Please select a preset by its number: ", False, "1", "2", "3", "4", "5"))
        if preset_in == 0:
            is_airsim_in = False
            reuse_in = False
            blue_in = 2
            red_in = 0
            episodes_in = 100000
            steps_in = 100
            paths_in = True
            obs_rate_in = 0.05
            type_in = 0
        elif preset_in == 1:
            is_airsim_in = False
            reuse_in = False
            blue_in = 2
            red_in = 1
            episodes_in = 100000
            steps_in = 100
            paths_in = True
            obs_rate_in = 0
            type_in = 1
        elif preset_in == 2:
            is_airsim_in = False
            reuse_in = False
            blue_in = 2
            red_in = 1
            episodes_in = 100000
            steps_in = 100
            paths_in = True
            obs_rate_in = 0
            type_in = 0
        elif preset_in == 3:
            is_airsim_in = False
            reuse_in = False
            blue_in = 2
            red_in = 2
            episodes_in = 100000
            steps_in = 100
            paths_in = True
            obs_rate_in = 0
            type_in = 0
        elif preset_in == 4:
            is_airsim_in = False
            reuse_in = False
            blue_in = 2
            red_in = 2
            episodes_in = 100000
            steps_in = 100
            paths_in = True
            obs_rate_in = 0.05
            type_in = 0

        training.training_loop(is_airsim_in, reuse_in, blue_in, red_in, episodes_in, steps_in, paths_in, obs_rate_in,
                               type_in)
        custom_set = True

    elif menu_in == "5":
        # viewer text
        viewer.text_view()

    elif menu_in == "6":
        # viewer airsim
        viewer.sim_view()

    elif menu_in == "7":
        print("To change variables such as learning settings, environment settings, and drone settings, see the readme")
        # edit environment settings
        # change what is possible, otherwise instruct
        # Agent velocity
        ...
    elif menu_in == "Q" or menu_in == "q":
        # quit
        print("Exiting program...")
        exit()
    else:
        print("Invalid Input - Select an option 1-6, or Q to quit")

print("Exiting program...")
