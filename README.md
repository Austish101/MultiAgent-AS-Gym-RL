This tool is designed for users with prior programming knowledge, and knowledge of reinforcement learning. 

Pre-requisites:
- AirSim and Unreal Engine, for running the drone simulations. https://microsoft.github.io/AirSim/#how-to-get-it 
- OpenAI Gym, needed for reinforcement learning environments. https://gym.openai.com/docs/
- Python with packages: gym, airsim, tensorflow. 


Menu:
To access the core of the tool, run Menu.py and use the text UI. 
- AirSim options refer to the simulation of drone agents, through Unreal Engine and the blocks environment. Ensure this environment is ready before running AirSim options. 
- Without simulation options do not simulate agents movements with drones, allowing for faster training. There is no visual to this during training, but the training session can be saved to view later.
- Custom settings allow for many options to be altered, recommended for use once the user is familiar with the tool. Detailed in the custom settings section. Reruns will be offered on the same instance. 
- Evalution mode allows for presets to be run. If a training run has been completed it will allow for evalutaion mode to be run. Presets detailed in Evaluation section. 
  - In evaluation mode learning is not continued over the same training course so the average steps taken over the otherwise same training run can be analysed. 
- View in text UI will allow for training runs that have been saved to be viewed using a text based UI. It allows for various episodes to be viewed, and displays a 2d representation of the 3d space for each step.
  - The colour represents what is in the space, the number represents the z value. Overlapping objects are possible.
- View in AirSim will rerun the saved training run in AirSim. 
- Changing environment settings is handled outside of the menu, refer to the section below. 
- Input validation is handled, and the user can enter Q to quit the program in most inputs. 


Custom settings:
- Learning agents - defines whether agents will be updated in this run. Differenciates training runs vs evaluation runs. 
- AirSim or Non-simulated environment
- Reuse Q-Tables - should the data for the agent be reused from the last saved training run. 
  - Allows for continuing agent training, or evaluation, or training with different settings. Q-tables are automatically saved after each run, to files Agent1/2/3/4. Previous data will be overwritten. 
- Number of agents - defines how many agents will be on each team. There must be one target finding agent. 
- Number of episodes - how many episodes should the run consist of. 
- Number of steps in episode - how many time steps should be allowed in each episode. 
  - Episode will end if a target finding agent reaches the goal, or if the number of steps in reached then the counter agents recieve a success reward.
 - Saving paths - Will the states of each step and episode be saved to paths.csv. This can then be used to view the run in the viewer. 
 - Obstacle rate - Only for non-simulated environments. User enters a percentage of randomly placed obstacles in the environment. They remaing for this training run only. 
   - For AirSim obstacles, the Unreal environment will need to be changed. 
   - Obstacles in the Unreal environment will only be viewed in post if the same environment is used when viewing using the view in AirSim option.
   - Obstacles in the non-simulated environment will only be viewable in the text UI viewer. 
- Moving destination - allows for the goal state to be moving during training, +y until the edge of the environment, then back using -y. 
- Agent types - Base allows all 'drones' to be agents and learn from experiences.
  - Moving obstacles changes the counter team to be moving predictably duing training, +y until the edge of the environment, then back using -y.
   

Change environment and config settings:
- To edit obstacles, or redesign the simulation, use the Unreal engine. Refer to the AirSim and Unreal engine documentation.
- config.ini
  - To change the play space of the simulation, goto the unreal_env section and change the settings of the x,y,z max and mins, and the cube size.
    - NOTE: cube size must be divisible by all max and mins, e.g. 5 given -20, 20, -10, 10 etc. 
    - NOTE: coordinates do not directly equate to Unreal coordinates, AirSim uses its own system based on the starting drone location. It also uses -z as upwards, and +z as downwards. 
  - To change the destinations used, goto unreal_env and change the coordinates for destinations. Must be within the play space defined above.
    - Recommended to set these coordinates to be in the middle of cube_size defined. 
  - If needed to edit drone settings, goto drone_agent where changes can be made to:
    - number of actions - set to 6 for +x, -x, +y, -y, +z, -z. Rework needed to add more actions. 
    - velocity of drones - NOTE: too high a speed can cause drones to crash due to the simulation of AirSim
    - timeout - how often will a moving drone check to see if it has collided with an obstacle. 
  - airsim_settings section is unused, but would be used to define the input of a camera on a drone.
  - learning_settings can be used to change:
    - learning rate of agents.
    - discount rate of agents.
    - the reward given to agents if their goal is complete.
    - the reward given to agents if they collide with an obstacle, or are otherwise blocked.
    - epsilon, defines the likelehood of exploring/exploting, i.e. choosing a random action, in epsilon greedy policy. 


Evaluation:
- At the end of each run, the time taken and the avergage steps over all episodes is displayed. 
- In text viewer, the steps for the selected episode is viewed.
- Presets in evaluation mode:
  - Basic Q-Learning, no sim, 2 targeting agents, environment obstacles, 100,000 episodes.
  - Basic Q-Learning, no sim, 2 targeting agents, 1 moving obstacle, 100,000 episodes.
  - Basic Q-Learning, no sim, 2 targeting agents, 1 learning drone, 100,000 episodes.
  - Basic Q-Learning, no sim, 2 targeting agents, 2 learning drones, 100,000 episodes.
  - Basic Q-Learning, no sim, 2 targeting agents, 2 learning drones, environment obstacles, 100,000 episodes.
- If no runs have been completed on the current instance of the tool when entering evaluation mode, these can be run with learning agents. 


AirSim:
  - AirSim settings can be changed in AirSim/settings.json, such as:
    - Number of drones initialsed, and the starting location.
    - Camera type: following, manual, etc.
  - NOTE: this will need to be edited for the correct amount of drones on startup. 
  
  
Files:
menu.py - Starting menu for training and viewing.
training.py - Runs using defined settings, training_loop() is the main function here calling to the rest where needed. 
viewer.py - Allows viewing using text UI or AirSim of previously saved runs. 
utility.py - Used to hold validation() function for input validation, used across multiple files. 
config.ini - Holds config settings, see the change environment and config settings section above. 
q_learning.py - QAgent class for q learning agents. 
AC_learning.py/DQN_learning.py - Partial implementation of Actor-Critic network and DQN reinforcement learning. 
  - NOTE: adding more learning techniques should be simple, users can use q_learning.py as a template, as long as new techniques hold:
    - update function - input observations, actions, rewards, done, etc.
    - get_action function - input observation, and output action.
    - save_data and read_data functions .
Agent1/2/3/4.csv - Holds agent data to be reused.
paths.csv - Holds the saved paths of agents through runs. 
env/AS_GymEnv.py - The Gym environment for AirSim simulation.
env/NoSim_GymEnv.py - Non simulated Gym environment.
env/drone_agent.py - The drone controller to connect to AirSim through the Gym environment. 
 
