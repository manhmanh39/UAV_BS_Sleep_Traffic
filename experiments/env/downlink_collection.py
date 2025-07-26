import copy
import numpy as np
import time
import yaml
# from pympler import tracker
from gym import spaces
import pandas as pd
import ast

#TODO:  - change from log-distance path loss to rician model (-> sinr, achievable rate)

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class DownlinkEnv(object):
    def __init__(self, data_dir="../../UAV_BS_Sleep_Traffic/generated_data/"):
        # self.tr = tracker.SummaryTracker()
        #load config
        self.config = load_config(data_dir + 'config.yaml')
        self.n_uav = self.config['total_uavs']
        self.grid_rows = self.config['grid_rows']
        self.grid_cols = self.config['grid_cols']
        self.cell_radius = self.config['cell_radius']
        self.max_timesteps = self.config['max_timesteps_per_episode']
        self.total_users = self.config['total_users']
        self.user_rate_range = self.config['user_rate_range']

        # load data
        self.traffic = pd.read_csv(data_dir + 'traffic.csv')
        self.bs_info = pd.read_csv(data_dir + 'bs_info.csv')
        self.bs_positions = self.bs_info[['x_pos', 'y_pos']].values
        self.episodes = pd.read_csv(data_dir + 'episodes.csv')

        #operational area bounds
        self.area_bounds = {
            'x_min': np.min(self.bs_positions[:, 0]) - self.cell_radius,
            'x_max': np.max(self.bs_positions[:, 0]) + self.cell_radius,
            'y_min': np.min(self.bs_positions[:, 1]) - self.cell_radius,
            'y_max': np.max(self.bs_positions[:, 1]) + self.cell_radius
        }

        self.user_info = pd.read_csv(data_dir + 'user_info.csv')
        self.user_positions = self.user_info[['x_pos', 'y_pos']].values
        self.user_demands = self.user_info['rate_requirement'].values
        self.user_cell_assignment = self.user_info['home_cell'].values


        # system parameters
        self.H = self.config['uav_altitude']  # height of UAVs
        self.v_max = self.confid['max_speed'] # max speed of UAVs
        self.P_max = self.config['max_power']  # max power of UAVs
        self.dt = self.config['time_step']  # time step
        self.B = self.config['bandwidth']  # bandwidth
        self.sigma_squared = self.config['noise_power']  # noise power
        self.PL_0 = self.config['path_loss_reference']  # path loss at reference distance
        self.alpha_pl = self.config['path_loss_exponent']  # path loss exponent

        #energy parameters
        self.alpha1 = self.config['propulsion_alpha1']  
        self.alpha2 = self.config['propulsion_alpha2']  

        #reward parameters
        self.w1 = self.config['coverage_weight'] #priority weight for coverage
        self.w2 = self.config['energy_weight']

        #energy normalization factor
        self.E_max = self.P_max * self.dt + self.alpha1 * (self.v_max ** 2) + self.alpha2 * self.v_max 

        #create grid for bs position (idf if needed)
        #initialize user positions and requirement

        #define action and observation spaces
        #action space: [vx, vy, power]
        self.action_space = [spaces.Box(low=np.array([-self.v_max, -self.v_max, 0]), 
                                        high=np.array([self.v_max, self.v_max, self.P_max]), 
                                        dtype=np.float32) for _ in range(self.n_uav)]
        
        # Observation: [own_pos(2), own_power(1), partner_pos(2), partner_power(1), 
        #               user_positions(200), user_demands(100), bs_states(K)]
        obs_dim = 2 + 1 + 2 + 1 + 200 + 100 + len(self.bs_info)
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), 
                                             dtype=np.float32)
                                             for _ in range(self.n_uav)]

        #initialize episode variables
        self.current_episode = 0
        self.timestep = 0
        self.cummulative_power = np.zeros(self.n_uav)
        self.force_return = np.array([False, False])

    def _get_bs_state(self, episode_idx, timestep):
        # Get the base station state for the given episode and timestep
        episode_data = self.episodes[self.episodes['episode_id'] == episode_idx]
        if len(episode_data) == 0:
            #default state if no data found (all is on)
            return np.ones(len(self.bs_info), dtype=int)  
        
        #get timestep data
        timestep_data = episode_data[episode_data['timestep'] == timestep]
        if len(timestep_data) == 0:
            #default state if no data found (all is on)
            return np.ones(len(self.bs_info), dtype=int)
        
        #extract bs states
        if 'bs_status' in timestep_data.columns:
            bs_states = timestep_data['bs_status'].values[0]
            bs_states = ast.literal_eval(bs_states) 
            return np.array(bs_states, dtype=int)
        else:
            raise ValueError("No 'bs_status' column in timestep data")
        

    def _calculate_distance(self, pos1, pos2):
        """
        Calculate the Euclidean distance between two positions.
        :param pos1: First position (x, y).
        :param pos2: Second position (x, y).
        :return: Distance between pos1 and pos2.
        """
        return np.sqrt(np.sum(np.square(np.array(pos1) - np.array(pos2))))
    
    def _calculate_path_loss(self, distance_2d):
        #log-distance path loss model with shadowing
        distance_3d = np.sqrt(distance_2d ** 2 + self.H ** 2)
        shadowing = np.random.normal(0, 6)  # up to 6db of shadowing
        return self.PL_0 + 10 * self.alpha_pl * np.log10(distance_3d)
    
    def _calculate_sinr(self, uav_idx, user_idx, uav_positions, uav_powers):
        user_pos = self.user_positions[user_idx]
        uav_pos = uav_positions[uav_idx]

        #pathloss_distance
        distance_2d = self._calculate_distance(uav_pos, user_pos)
        path_loss_db = self._calculate_path_loss(distance_2d)
        path_loss_linear = 10 ** (path_loss_db / 10)

        #signal power from the UAV
        signal_power = uav_powers[uav_idx] / path_loss_linear

        #interference from other UAVs
        interference = 0
        for other_uav in range(self.n_uav):
            if other_uav != uav_idx:
                other_distance_2d = self._calculate_distance(uav_positions[other_uav], user_pos)
                other_path_loss_db = self._calculate_path_loss(other_distance_2d)
                other_path_loss_linear = 10 ** (other_path_loss_db / 10)
                interference += uav_powers[other_uav] / other_path_loss_linear

        #interference from BSs
        user_cell = self.user_cell_assignment[user_idx]
        for bs_idx in range(self.bs_positions):
            if self.bs_states[bs_idx] == 1: #bs on
                bs_distance_2d = self._calculate_distance(self.bs_positions[bs_idx], user_pos)
                bs_path_loss_db = self._calculate_path_loss(bs_distance_2d)
                bs_path_loss_linear = 10 ** (bs_path_loss_db / 10)
                #assume bs power is constant and equal to P_max
                interference += self.P_max / bs_path_loss_linear

        #calculate SINR
        sinr = signal_power / (interference + self.sigma_squared)
        return sinr

    def _calculate_achievable_rate(self,uav_idx, user_idx, uav_positions, uav_powers):
        sinr = self._calculate_sinr(uav_idx, user_idx, uav_positions, uav_powers)
        if sinr <= 0:
            return 0
        return self.B * np.log2(1 + sinr)
    
    def _get_active_users(self):
        #get users that need uav 
        active_users = []
        for user_idx in range(len(self.user_positions)):
            user_cell = self.user_cell_assignment[user_idx]
            if self.bs_states[user_cell] == 0:  # BS is offf
                active_users.append(user_idx)

        return np.array(active_users)
    

    def _assign_users_to_uavs(self, uav_positions, uav_powers):
        #assign to highest sinr
        active_users = self._get_active_users()
        user_assignments = {}
        for user_idx in active_users:
            best_uav = -1
            best_sinr = -1

            for uav_idx in range(self.n_uav):
                sinr = self._calculate_sinr(uav_idx, user_idx, uav_positions, uav_powers)
                if sinr > best_sinr:
                    best_sinr = sinr
                    best_uav = uav_idx

            if best_uav != -1:
                user_assignments[user_idx] = best_uav

        return user_assignments
    
    def _calculate_coverage_ratio(self, uav_idx, uav_positions, uav_powers):
        #only count if rate requirement is met
        user_assignments = self._assign_users_to_uavs(uav_positions, uav_powers)
        active_users = self._get_active_users()

        if len(active_users) == 0:
            return 0.0
        
        served_users = 0
        for user_idx, assigned_uav in user_assignments.items():
            if assigned_uav == uav_idx:
                rate = self._calculate_achievable_rate(uav_idx, user_idx, uav_positions, uav_powers)
                if rate >= self.user_demands[user_idx]:
                    served_users += 1

        return served_users / len(active_users)
    
    def _calculate_energy_consumption(self, uav_idx, delta_position, power):
        #communicaion energy
        E_comm = power * self.dt
        #propulsion energy
        speed = np.linalg.norm(delta_position) 
        E_prop = self.alpha1 * (speed ** 2) + self.alpha2 * speed

        return E_comm + E_prop
    
    def _calculate_reward(self, uav_idx, uav_positions, uav_powers, delta_position):
        """calculate reward for single uav"""
        #coverage ratio
        coverage_ratio = self._calculate_coverage_ratio(uav_idx, uav_positions, uav_powers)

        #energy openalty
        energy_consumption = self._calculate_energy_consumption(uav_idx, delta_position, uav_powers[uav_idx])
        energy_penalty = energy_consumption / self.E_max  # normalize to [0, 1]
        #reward
        reward = self.w1 * coverage_ratio - self.w2 * energy_penalty

        return reward

    def _get_observation(self, uav_idx):
        obs = []
        #own position and power
        obs.extend(self.uav_positions[uav_idx])  # (x, y)
        obs.append(self.uav_powers[uav_idx])  # power
        #partner positions and powers
        partner_idx = 1 - uav_idx  # 2 UAVs
        obs.extend(self.uav_positions[partner_idx])  # (x, y)
        obs.append(self.uav_powers[partner_idx])  
        #all user positions (flattened)
        obs.extend(self.user_positions.flatten())  # (x1, y1, x2, y2, ...)
        #all user demands
        obs.extend(self.user_demands)
        #bs states
        obs.extend(self.bs_states)

        return np.array(obs, dtype=np.float32)
    
    def _check_boundaries(self, position):
        x,y = position
        return (
            self.area_bounds['x_min'] <= x <= self.area_bounds['x_max'] and
            self.area_bounds['y_min'] <= y <= self.area_bounds['y_max']
        )

    def reset(self):
        self.timestep = 0
        self.cummulative_power = np.zeros(self.n_uav)
        self.force_return = np.array([False, False])

        #initialize UAV positions 
        self.uav_positions = np.array([[0, 0], [0, 0]]) #shape (n_uav,2)
        self.final_uav_positions = np.array([[0, 0], [0, 0]])

        #initialize UAV powers
        self.uav_powers = np.zeros(self.n_uav)

        #get initial BS states
        self.bs_states = self._get_bs_state(self.current_episode, self.timestep)

        #get inital observations
        obs = [self._get_observation(i) for i in range(self.n_uav)]

        return obs

    def step(self, actions):
        #process actions
        new_positions = []
        new_powers = []
        delta_positions = []

        for i, action in enumerate(actions):
            #action extract
            delta_x, delta_y, power = action
            
            #apply contstraints
            delta_position = np.array([delta_x, delta_y])
            speed = np.linalg.norm(delta_position)
            if speed > self.v_max:
                delta_position = (delta_position / speed) * self.v_max

            power = np.clip(power, 0, self.P_max)

            #update cummulative power for each uav
            self.cummulative_power[i] += power
            if self.cummulative_power >= self.P_max:
                self.force_return[i] = True
                continue

            #update position
            new_position = self.uav_positions[i] + delta_position
            if self._check_boundaries(new_position):
                new_positions.append(new_position)
            else:
                #if out of bounds, keep old position
                new_positions.append(self.uav_positions[i])
                delta_position = np.array([0, 0])  # no movement

            new_powers.append(power)
            delta_positions.append(delta_position)

        #update UAV positions and powers
        for i in range(len(self.n_uav)):
            if self.force_return[i] == True:
                self.uav_positions = self.final_uav_positions.copy()
                self.uav_powers = np.zeros(self.n_uav)
            else:
                self.uav_positions = np.array(new_positions)
                self.uav_powers = np.array(new_powers)

        #update BS states
        self.bs_states = self._get_bs_state(self.current_episode, self.timestep)

        #calculate rewards
        rewards = []
        for i in range(self.n_uav):
            reward = self._calculate_reward(i, self.uav_positions, self.uav_powers, delta_positions[i])
            rewards.append(reward)

        #check if episode is done
        done = self.timestep >= self.max_timesteps
        if np.all(self.force_return):
            done = True
        if done:
            self.current_episode += 1
            self.timestep = 0
        else:
            self.timestep += 1

        #get new observations
        obs = [self._get_observation(i) for i in range(self.n_uav)]
        #info dictionary
        info = {
            'coverage_ratios': [self._calculate_coverage_ratio(i, self.uav_positions, self.uav_powers) for i in range(self.n_uav)],
            'active_users': len(self._get_active_users()),
            'timestep': self.timestep,
        }

        return obs, rewards, done, info


    def render(self, mode='human'):
        #(optional
        print(f"Timestep: {self.timestep}")
        print(f"UAV Positions: {self.uav_positions}")
        print(f"UAV Powers: {self.uav_powers}")
        print(f"BS States: {self.bs_states}")
        print(f"Active Users: {len(self._get_active_users())}")
        print("-" * 50)

    def close(self):
        # (optional)
        pass

    @property
    def n(self):
        return self.n_uav