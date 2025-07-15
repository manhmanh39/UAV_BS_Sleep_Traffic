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

    # Thêm randomization mạnh: nhân thêm phân phối ngẫu nhiên
    random_factor = np.random.exponential(scale=1.0, size=timesteps)  # Exponential để tạo nhiều giá trị nhỏ và một số giá trị lớn bất thường
    # Hoặc nếu muốn đơn giản hơn, dùng uniform cũng được:
    # random_factor = np.random.uniform(0.5, 1.5, size=timesteps)

    weights = (1 + seasonal + trend + noise) * random_factor
    weights = np.clip(weights, 0, None)
    return weights

def generate_traffic(raw_data_dir, output_dir, grid_rows, grid_cols):
    raw_data_files = get_all_raw_data_files(raw_data_dir)
    total_dl = load_total_DL(raw_data_files)
    max_timesteps = len(total_dl)

    n_bs = grid_rows * grid_cols

    # Precompute weights for all BSs
    all_bs_weights = [traffic_weaver_weights(max_timesteps, bs_id, n_bs) for bs_id in range(n_bs)]

    traffic_data = []
    for t in range(max_timesteps):
        row = {'timestep': int(t)}

        weights = np.array([all_bs_weights[bs_id][t] for bs_id in range(n_bs)])
        weights = weights / np.sum(weights)

        for bs_id in range(n_bs):
            dl = float(total_dl[t] * weights[bs_id])
            row[f'BS_{bs_id}_DL'] = dl

        traffic_data.append(row)

    df = pd.DataFrame(traffic_data)
    df.to_csv(os.path.join(output_dir, 'traffic.csv'), index=False)
    print("✅ traffic.csv generated")

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

def generate_episodes(output_dir, grid_rows, grid_cols, traffic_file, max_timesteps_per_episode, max_episodes, total_uavs):
    n_bs = grid_rows * grid_cols
    traffic = pd.read_csv(traffic_file)

    episodes = []

    for ep in range(max_episodes):
        for t in range(max_timesteps_per_episode):
            global_timestep = (ep * max_timesteps_per_episode + t) % len(traffic)

            # Lấy traffic tại timestep hiện tại
            total_DL = sum([traffic.loc[global_timestep, f'BS_{bs}_DL'] for bs in range(n_bs)])
            bs_status = []
            for bs in range(n_bs):
                bs_dl = traffic.loc[global_timestep, f'BS_{bs}_DL']
                
                # Ngưỡng 1% trung bình
                threshold = 0.1 * (total_DL / n_bs)
                if bs_dl < threshold:
                    bs_status.append(0)  # Sleep
                else:
                    bs_status.append(1)  # Active

            # Chỉ sinh UAV ở timestep=0 của mỗi episode
            if t == 0:
                uav_positions = [[float(np.random.uniform(0, grid_cols * 1.0)), 
                                  float(np.random.uniform(0, grid_rows * 1.0))] for _ in range(total_uavs)]
            else:
                uav_positions = None

            episodes.append({
                'episode_id': int(ep),
                'timestep': int(t),
                'bs_status': bs_status,
                'uav_init_positions': uav_positions
            })

    df = pd.DataFrame(episodes)
    df.to_csv(os.path.join(output_dir, 'episodes.csv'), index=False)
    print("✅ episodes.csv generated with BS sleep mode based on traffic threshold")

def save_config(output_dir, config_dict):
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)
    print("✅ config.yaml saved")

if __name__ == "__main__":
    print("🔄 Starting data generation...")

    config = {
        'grid_rows': 3,
        'grid_cols': 3,
        'cell_radius': 0.5,
        'total_users': 100,
        'total_uavs': 2,
        'user_rate_range': [0.5, 5.0],
        'max_episodes': 100,
        'max_timesteps_per_episode': 500
    }

    max_timesteps = generate_traffic(raw_data_dir, output_dir, config['grid_rows'], config['grid_cols'])
    generate_bs(output_dir, config['grid_rows'], config['grid_cols'], config['cell_radius'])
    generate_users(output_dir, config['grid_rows'], config['grid_cols'], config['cell_radius'], config['total_users'], config['user_rate_range'])
    
    generate_episodes(
        output_dir, 
        config['grid_rows'], 
        config['grid_cols'], 
        os.path.join(output_dir, 'traffic.csv'),
        config['max_timesteps_per_episode'], 
        config['max_episodes'], 
        config['total_uavs']
    )

    save_config(output_dir, config)

    print("✅ Data generation completed successfully!")