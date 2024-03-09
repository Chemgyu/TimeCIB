import numpy as np

time_length = 48
data_dim = 35
data = np.load(f"data/physionet/raw/physionet.npz")

x_train_full = data['x_train_full']
x_train_miss = data['x_train_miss']
m_train_miss = data['m_train_miss']
m_train_artificial = data["m_train_artificial"]
y_train = data['y_train'].reshape([-1, 1])
t_train = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_train_full), axis=0)

x_valid_full = data['x_val_full']
m_valid_miss = data['m_val_miss']
x_valid_miss = data['x_val_miss']
m_valid_artificial = data["m_val_artificial"]
y_valid = data['y_val'].reshape([-1, 1])
t_valid = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_valid_full), axis=0)

x_test_full = data['x_test_full']
x_test_miss = data['x_test_miss']
m_test_miss = data['m_test_miss']
m_test_artificial = data["m_test_artificial"]
y_test = data['y_test'].reshape([-1, 1])
t_test = np.repeat(np.expand_dims(np.arange(0, time_length), 0), len(x_test_full), axis=0)

print(f"saving data/physionet/physionet.npz ...")
np.savez(f"data/physionet/physionet.npz", 
        x_train_full=np.array(x_train_full), x_train_miss=np.array(x_train_miss), m_train_miss=np.array(m_train_miss), m_train_artificial=np.array(m_train_artificial), y_train=np.array(y_train), t_train=np.array(t_train), 
        x_valid_full=np.array(x_valid_full), x_valid_miss=np.array(x_valid_miss), m_valid_miss=np.array(m_valid_miss), m_valid_artificial=np.array(m_valid_artificial), y_valid=np.array(y_valid), t_valid=np.array(t_valid),
        x_test_full=np.array(x_test_full),   x_test_miss =np.array(x_test_miss),  m_test_miss =np.array(m_test_miss),  m_test_artificial =np.array(m_test_artificial) , y_test =np.array(y_test) , t_test=np.array(t_test)
)
