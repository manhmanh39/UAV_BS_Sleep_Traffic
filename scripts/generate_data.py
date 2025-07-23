import numpy as np
import pandas as pd
import os
import yaml

np.random.seed(42)  # For reproducibility

# Set absolute paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raw_data_dir = os.path.join(project_root, 'raw_data')
output_dir = os.path.join(project_root, 'generated_data')

os.makedirs(output_dir, exist_ok=True)
print(f"✅ Project Root: {project_root}")

def get_all_raw_data_files(raw_data_dir):
    raw_files = [os.path.join(raw_data_dir, file) 
                 for file in os.listdir(raw_data_dir) 
                 if file.endswith('_dataset.csv')]
    assert len(raw_files) > 0, "❌ No raw data files found!"
    return raw_files

def load_total_DL(raw_data_files):
    total_dl = None
    for file in raw_data_files:
        df = pd.read_csv(file)
        dl_cols = [col for col in df.columns if 'DL_bitrate' in col]
        assert len(dl_cols) > 0, f"❌ No DL_bitrate column in {file}"
        dl_sum = df[dl_cols].sum(axis=1)

        if total_dl is None:
            total_dl = dl_sum
        else:
            total_dl += dl_sum

    return total_dl.values

def traffic_weaver_weights(timesteps, bs_id, total_bs, seasonal_scale=0.2, trend_scale=0.05, noise_scale=0.05):
    t = np.arange(timesteps)
    seasonal = seasonal_scale * np.sin(2 * np.pi * t / 24 + bs_id)
    trend = trend_scale * (bs_id / total_bs) * t / timesteps
    noise = np.random.normal(0, noise_scale, timesteps)

    random_factor = np.random.exponential(scale=1.0, size=timesteps)  # Exponential để tạo nhiều giá trị nhỏ và một số giá trị lớn bất thường
    
    weights = (1 + seasonal + trend + noise) * random_factor
    weights = np.clip(weights, 0, None)
    return weights

def generate_traffic(raw_data_dir, output_dir, grid_rows, grid_cols):
    raw_data_files = get_all_raw_data_files(raw_data_dir)
    total_dl = load_total_DL(raw_data_files)
    max_timesteps = len(total_dl)

    n_bs = grid_rows * grid_cols

    # Khởi tạo hotness gốc (hotspot/coldspot)
    base_hotness = np.random.uniform(0.5, 2.0, n_bs)
    hotspot_indices = np.random.choice(n_bs, size=int(0.3 * n_bs), replace=False)
    for idx in hotspot_indices:
        base_hotness[idx] = np.random.uniform(3.0, 4.0)  # Hotspot cơ bản

    def daily_pattern(t):
        hour = t % 24
        if 7 <= hour <= 9 or 18 <= hour <= 21:
            return 1.5  # Giờ cao điểm
        elif 0 <= hour <= 5:
            return 0.5  # Ban đêm
        else:
            return 1.0

    def event_spike(prob=0.05):
        return np.random.choice([1.0, np.random.uniform(1.2, 2.0)], p=[1-prob, prob])

    traffic_data = []
    for t in range(max_timesteps):
        row = {'timestep': int(t)}
        hour = t % 24

        dynamic_hotness = np.copy(base_hotness)

        # Ban đêm: hotspot giảm tải 50%
        if 0 <= hour <= 5:
            for idx in hotspot_indices:
                dynamic_hotness[idx] *= 0.5

        # Coldspot có 10% cơ hội tăng đột biến để kích hoạt UAV/coldspot bật lên
        coldspot_indices = [i for i in range(n_bs) if i not in hotspot_indices]
        for idx in coldspot_indices:
            if np.random.rand() < 0.1:
                dynamic_hotness[idx] *= np.random.uniform(1.5, 3.0)

        # Chuẩn hóa lại hotness bằng softmax mềm
        soft_hotness = np.exp(dynamic_hotness) / np.sum(np.exp(dynamic_hotness))

        for bs_id in range(n_bs):
            base_traffic = total_dl[t] * soft_hotness[bs_id]
            traffic = base_traffic * daily_pattern(t) * event_spike() * (1 + np.random.normal(0, 0.05))
            traffic = max(traffic, 0.0)
            row[f'BS_{bs_id}_DL'] = float(traffic)

        traffic_data.append(row)

    df = pd.DataFrame(traffic_data)
    df.to_csv(os.path.join(output_dir, 'traffic.csv'), index=False)
    print("✅ Balanced traffic.csv generated with hotspot/coldspot dynamic")

    return max_timesteps

def generate_bs(output_dir, grid_rows, grid_cols, cell_radius):
    bs_list = []
    bs_id = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * cell_radius * 2
            y = row * cell_radius * 2
            bs_list.append({
                'bs_id': int(bs_id),
                'x_pos': float(x),
                'y_pos': float(y),
                'cell_radius': float(cell_radius),
                'row': int(row),
                'col': int(col)
            })
            bs_id += 1

    df = pd.DataFrame(bs_list)
    df.to_csv(os.path.join(output_dir, 'bs_info.csv'), index=False)
    print("✅ bs_info.csv generated")

def generate_users(output_dir, grid_rows, grid_cols, cell_radius, total_users, user_rate_range):
    bs_pos = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * cell_radius * 2
            y = row * cell_radius * 2
            bs_pos.append((x, y))

    users_list = []
    for user_id in range(total_users):
        x = np.random.uniform(0, grid_cols * cell_radius * 2)
        y = np.random.uniform(0, grid_rows * cell_radius * 2)
        rate_req = np.random.uniform(*user_rate_range)
        dists = [np.linalg.norm([x - bsx, y - bsy]) for bsx, bsy in bs_pos]
        home_bs = np.argmin(dists)
        user_type = np.random.choice(['voice', 'data', 'video'], p=[0.3, 0.5, 0.2])

        users_list.append({
            'user_id': int(user_id),
            'x_pos': float(x),
            'y_pos': float(y),
            'rate_requirement': float(rate_req),
            'home_cell': int(home_bs),
            'user_type': user_type
        })

    df = pd.DataFrame(users_list)
    df.to_csv(os.path.join(output_dir, 'user_info.csv'), index=False)
    print("✅ user_info.csv generated")

def generate_uav_home_positions(output_dir, grid_rows, grid_cols, total_episodes, total_uavs, cell_radius):
    uav_data = []
    home_point = (0, 0)  # Starting point for UAVs
    for ep in range(total_episodes):
        for uav_id in range(total_uavs):
            x,y = home_point
            uav_data.append({
                'episode_id': int(ep),
                'uav_id': int(uav_id),
                'x_init': x,
                'y_init': y,
                'x_final': x,
                'y_final': y,
            })
    df = pd.DataFrame(uav_data)
    df.to_csv(os.path.join(output_dir, 'uav_home_positions.csv'), index=False)
    print("✅ uav_home_positions.csv generated")

def generate_episodes(output_dir, grid_rows, grid_cols, traffic_file, max_timesteps_per_episode, max_episodes, cell_radius):
    n_bs = grid_rows * grid_cols
    traffic = pd.read_csv(traffic_file)

    bs_means = {bs: traffic[f'BS_{bs}_DL'].mean() for bs in range(n_bs)}
    episodes = []

    for ep in range(max_episodes):
        for t in range(max_timesteps_per_episode):
            global_timestep = (ep * max_timesteps_per_episode + t) % len(traffic)
            bs_status = []
            for bs in range(n_bs):
                bs_dl = traffic.loc[global_timestep, f'BS_{bs}_DL']
                threshold = 0.5 * bs_means[bs]
                bs_status.append(1 if bs_dl >= threshold else 0)

            episodes.append({
                'episode_id': int(ep),
                'timestep': int(t),
                'bs_status': bs_status
            })

    df = pd.DataFrame(episodes)
    df.to_csv(os.path.join(output_dir, 'episodes.csv'), index=False)
    print("✅ episodes.csv generated without UAV positions")

def save_config(output_dir, config_dict):
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)
    print("✅ config.yaml saved")

if __name__ == "__main__":
    print("🔄 Starting data generation...")
    config = {
        'grid_rows': 3,
        'grid_cols': 3,
        'cell_radius': 500,
        'total_users': 100,
        'total_uavs': 2,
        'user_rate_range': [0.5, 5.0],
        'max_episodes': 100,
        'max_timesteps_per_episode': 500
    }

    max_timesteps = generate_traffic(raw_data_dir, output_dir, config['grid_rows'], config['grid_cols'])
    generate_bs(output_dir, config['grid_rows'], config['grid_cols'], config['cell_radius'])
    generate_users(output_dir, config['grid_rows'], config['grid_cols'], config['cell_radius'], config['total_users'], config['user_rate_range'])
    generate_uav_home_positions(output_dir, config['grid_rows'], config['grid_cols'], config['max_episodes'], config['total_uavs'], config['cell_radius'])
    generate_episodes(output_dir, config['grid_rows'], config['grid_cols'], os.path.join(output_dir, 'traffic.csv'), config['max_timesteps_per_episode'], config['max_episodes'], config['cell_radius'])
    save_config(output_dir, config)

    print("✅ Data generation completed successfully!")
