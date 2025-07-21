import copy
import numpy as np
import time
import yaml
# from pympler import tracker
from gym import spaces
import pandas as pd
import ast

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


    def reset(self):
        # initialize data map
        # tr = tracker.SummaryTracker()
        self.mapmatrix = copy.copy(self._mapmatrix)
        # ---- original
        # self.maptrack = np.zeros(self.mapmatrix.shape)
        # ---- new 6-7-11-28
        # self.maptrack = np.ones(self.mapmatrix.shape) * self.track
        # ---- 18:43
        self.maptrack = np.zeros(self.mapmatrix.shape)
        # ----
        # initialize positions of uavs
        self.uav = [list(self.sg.V['INIT_POSITION']) for i in range(self.n)]
        self.eff = [0.] * self.n
        self.count = 0
        self.zero = 0

        self.trace = [[] for i in range(self.n)]
        # initialize remaining energy
        self.energy = np.ones(self.n).astype(np.float64) * self.maxenergy
        # initialize indicators
        self.collection = np.zeros(self.n).astype(np.float16)
        self.walls = np.zeros(self.n).astype(np.int16)

        # time
        self.time_ = 0

        # initialize images
        self.state = self.__init_image()
        # print(self.fairness)
        # image = [np.reshape(np.array([self.image_data, self.image_position[i]]), (self.map.width, self.map.height, self.channel)) for i in range(self.n)]
        # tr.print_diff()
        return self.__get_state()

    def __get_eff(self, value, distance):
        return value / self.maxenergy

    def __get_eff1(self, value, distance):
        return value / (distance + self.alpha * value + self.epsilon)

    def __cusume_energy0(self, uav, value, distance):
        self.energy[uav] -= distance

    def __cusume_energy1(self, uav, value, distance, energy=None):
        if energy is None:
            # ---- or
            # self.erengy[uav] -= (distance + value * self.alpha)
            # ---- 6-8 14:48
            self.energy[uav] -= (self.factor * distance + self.alpha * value)
            # ----
        else:
            # ---- or
            # energy[uav] -= (distance + value * self.alpha)
            # ---- 14:48
            energy[uav] -= (self.factor * distance + self.alpha * value)
            # ----

    # ---- or
    # def step(self, action):
    #     self.count += 1
    #     actions = copy.deepcopy(action)
    #     normalize = self.normalize
    #     for i in range(self.n):
    #         for ii in actions[i]:
    #             if np.isnan(ii):
    #                 print('Nan')
    #                 while True:
    #                     pass
    # ---- 6-8 10:57
    def step(self, actions,indicator=None):
        self.count += 1
        action = copy.deepcopy(actions)
        # 6-20 00:43
        if np.max(action) > self.maxaction:
            self.maxaction = np.max(action)
            # print(self.maxaction)
        if np.min(action) < self.minaction:
            self.minaction = np.min(action)
            # print(self.minaction)
        action = np.clip(action, -1e3, 1e3)  #

        normalize = self.normalize

        #TODO:梯度爆炸问题不可小觑,
        # 遇到nan直接卡掉
        for i in range(self.n):
            for ii in action[i]:
                if np.isnan(ii):
                    print('Nan')
                    while True:
                        pass

        reward = [0] * self.n
        self.dn = [False] * self.n  # no energy UAV
        update_points = []
        update_tracks = []
        clear_uav = copy.copy(self.uav)
        new_positions = []
        c_f = self.__get_fairness(self.maptrack)
        # update positions of UAVs
        for i in range(self.n):
            self.trace[i].append(self.uav[i])
            distance = np.sqrt(np.power(action[i][0], 2) + np.power(action[i][1], 2))
            data = 0

            if distance <= self.maxdistance and self.energy[i] >= distance:
                new_x = self.uav[i][0] + action[i][0]
                new_y = self.uav[i][1] + action[i][1]
            else:
                maxdistance = self.maxdistance if self.maxdistance <= self.energy[i] else self.energy[i]
                if distance <= self.epsilon:
                    distance = self.epsilon
                    print("very small.")
                new_x = self.uav[i][0] + maxdistance * action[i][0] / distance
                new_y = self.uav[i][1] + maxdistance * action[i][1] / distance
                distance = maxdistance
            if distance <= self.epsilon:
                self.zero += 1
            self.__cusume_energy1(i, 0, distance)
            if 0 <= new_x < self.mapx and 0 <= new_y < self.mapy and self.mapob[myint(new_x)][myint(new_y)] != self.OB:
                new_positions.append([new_x, new_y])
            else:
                new_positions.append([self.uav[i][0], self.uav[i][1]])
                reward[i] += normalize * self.pwall
                self.walls[i] += 1
            # calculate distances between UAV and data points
            _pos = np.repeat([new_positions[-1]], [self.datas.shape[0]], axis=0)
            _minus = self.datas - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)
            for index, dis in enumerate(_dis):
                if np.sqrt(dis) <= self.crange:
                    self.maptrack[index] += self.track
                    update_tracks.append([index, self.maptrack[index]])
                    if self.mapmatrix[index] > 0:
                        data += self._mapmatrix[index] * self.cspeed
                        self.mapmatrix[index] -= self._mapmatrix[index] * self.cspeed
                        if self.mapmatrix[index] < 0:
                            self.mapmatrix[index] = 0.
                        update_points.append([index, self.mapmatrix[index]])
            # update info
            value = data if self.energy[i] >= data * self.alpha else self.energy[i]
            self.__cusume_energy1(i, value, 0.)
            c_f_ = self.__get_fairness(self.maptrack)
            # ---- 6-7
            # 11:32
            # reward[i] += self.__get_reward9(value, distance, c_f, c_f_)
            # ---- 11:44
            # reward[i] += self.__get_reward7(value, distance, c_f, c_f_)
            # ---- 18:43
            reward[i] += self.__get_reward9(value, distance, c_f, c_f_)
            # ----
            c_f = c_f_
            self.eff[i] += self.__get_eff1(value, distance)
            self.collection[i] += value
            if self.energy[i] <= self.epsilon * self.maxenergy:
                self.dn[i] = True
        self.uav = new_positions
        t = time.time()
        self.__draw_image(clear_uav, update_points, update_tracks)
        self.time_ += time.time() - t
        # ---- or
        reward = list(np.clip(np.array(reward) / normalize, -2., 1.))
        # ---- new 18:43
        # reward = list(np.clip(np.array(reward) / normalize, -1., 1.))
        # ---- end new
        info = None
        state = self.__get_state()
        for r in reward:
            if np.isnan(r):
                print('Rerward Nan')
                while True:
                    pass
        return state, reward, sum(self.dn), info,indicator

    def render(self):
        print('coding...')

    @property
    def leftrewards(self):
        return np.sum(self.mapmatrix) / self.totaldata

    @property
    def efficiency(self):
        return np.sum(self.collection / self.totaldata) / (
                    self.n - np.sum(self.normal_energy)) * self.collection_fairness

    @property
    def normal_energy(self):
        return list(np.array(self.energy) / self.maxenergy)

    @property
    def fairness(self):
        square_of_sum = np.square(np.sum(self.mapmatrix[:]))
        sum_of_square = np.sum(np.square(self.mapmatrix[:]))
        fairness = square_of_sum / sum_of_square / float(len(self.mapmatrix))
        return fairness

    @property
    def collection_fairness(self):
        collection = self._mapmatrix - self.mapmatrix
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness

    @property
    def normal_collection_fairness(self):
        collection = self._mapmatrix - self.mapmatrix
        for index, i in enumerate(collection):
            collection[index] = i / self._mapmatrix[index]
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness
