import numpy as np

def apply_missing(data, missing_ratio, mask=None):
    data = np.array(data)
    if mask==None: mask = np.random.random(size=data.shape) < missing_ratio
    data[mask] = 0
    return data, mask


time_length = 10
names = ["hmnist_mnar.npz", "hmnist_random.npz", "hmnist_spatial.npz", "hmnist_temporal_neg.npz", "hmnist_temporal_pos.npz"]
val_split = 50000
for dataname in names:
        data = np.load(f"data/hmnist/raw/{dataname}")

        x_train_full = data['x_train_full'][:val_split]
        x_train_miss = data['x_train_miss'][:val_split]
        m_train_miss = data['m_train_miss'][:val_split]
        y_train = data['y_train'][:val_split]
        t_train = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_train_full), axis=0)

        x_valid_full = data['x_train_full'][val_split:]
        x_valid_miss = data['x_train_miss'][val_split:]
        m_valid_miss = data['m_train_miss'][val_split:]
        y_valid = data['y_train'][val_split:]
        t_valid = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_valid_full), axis=0)

        x_test_full = data['x_test_full']
        x_test_miss = data['x_test_miss']
        m_test_miss = data['m_test_miss']
        y_test = data['y_test']
        t_test = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_test_full), axis=0)

        print(f"saving data/hmnist/{dataname}...")
        np.savez(f"data/hmnist/{dataname}", 
                x_train_full=np.array(x_train_full), x_train_miss=np.array(x_train_miss), m_train_miss=np.array(m_train_miss), m_train_artificial=np.array(m_train_miss), y_train=np.array(y_train), t_train=t_train,
                x_valid_full=np.array(x_valid_full), x_valid_miss=np.array(x_valid_miss), m_valid_miss=np.array(m_valid_miss), m_valid_artificial=np.array(m_valid_miss), y_valid=np.array(y_valid), t_valid=t_valid,
                x_test_full=np.array(x_test_full),   x_test_miss =np.array(x_test_miss),  m_test_miss =np.array(m_test_miss),  m_test_artificial=np.array(m_test_miss)  ,  y_test=np.array(y_test),   t_test=t_test
                )

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for ratio in ratios:
        data = np.load("data/hmnist/raw/hmnist_random.npz")

        x_train_full = data['x_train_full'][:val_split]
        x_train_miss, m_train_miss = apply_missing(x_train_full, ratio)
        y_train = data['y_train'][:val_split]
        t_train = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_train_full), axis=0)

        x_valid_full = data['x_train_full'][val_split:]
        x_valid_miss, m_valid_miss = apply_missing(x_valid_full, ratio)
        y_valid = data['y_train'][val_split:]
        t_valid = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_valid_full), axis=0)

        x_test_full = data['x_test_full']
        x_test_miss, m_test_miss = apply_missing(x_test_full, ratio)
        y_test = data['y_test']
        t_test = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_test_full), axis=0)

        print(f"saving data/hmnist/hmnist_random_{ratio}...")
        np.savez(f"data/hmnist/hmnist_random_{ratio}.npz", 
                x_train_full=np.array(x_train_full), x_train_miss=np.array(x_train_miss), m_train_miss=np.array(m_train_miss), m_train_artificial=np.array(m_train_miss), y_train=np.array(y_train), t_train=t_train,
                x_valid_full=np.array(x_valid_full), x_valid_miss=np.array(x_valid_miss), m_valid_miss=np.array(m_valid_miss), m_valid_artificial=np.array(m_valid_miss), y_valid=np.array(y_valid), t_valid=t_valid,
                x_test_full=np.array(x_test_full),   x_test_miss =np.array(x_test_miss),  m_test_miss =np.array(m_test_miss),  m_test_artificial=np.array(m_test_miss)  ,  y_test=np.array(y_test),   t_test=t_test
                )