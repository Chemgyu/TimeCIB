# This code is adopted from GP-VAE: https://github.com/ratschlab/GP-VAE/blob/master/data/load_hmnist.sh
# Each dataset consumes ~ 6.1 GB. (Total 30 GB for five datasets)

DATA_DIR="data/raw/hmnist"
mkdir -p ${DATA_DIR}

wget https://www.dropbox.com/s/xzhelx89bzpkkvq/hmnist_mnar.npz?dl=1 -O ${DATA_DIR}/hmnist_mnar.npz
wget https://www.dropbox.com/s/jiix44usv7ibv1z/hmnist_spatial.npz?dl=1 -O ${DATA_DIR}/hmnist_spatial.npz
wget https://www.dropbox.com/s/7s5y70f4idw9nei/hmnist_random.npz?dl=1 -O ${DATA_DIR}/hmnist_random.npz
wget https://www.dropbox.com/s/fnqi4rv9wtt2hqo/hmnist_temporal_neg.npz?dl=1 -O ${DATA_DIR}/hmnist_temporal_neg.npz
wget https://www.dropbox.com/s/tae3rdm9ouaicfb/hmnist_temporal_pos.npz?dl=1 -O ${DATA_DIR}/hmnist_temporal_pos.npz

## Use below code if you want to download only one missing types.

# random_mechanism="mnar"
# if [ "$random_mechanism" == "mnar" ] ; then
#     wget https://www.dropbox.com/s/xzhelx89bzpkkvq/hmnist_mnar.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
# elif [ "$random_mechanism" == "spatial"] ; then
#     wget https://www.dropbox.com/s/jiix44usv7ibv1z/hmnist_spatial.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
# elif [ "$random_mechanism" == "random" ] ; then
#     wget https://www.dropbox.com/s/7s5y70f4idw9nei/hmnist_random.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
# elif [ "$random_mechanism" == "temporal_neg" ] ; then
#     wget https://www.dropbox.com/s/fnqi4rv9wtt2hqo/hmnist_temporal_neg.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
# elif [ "$random_mechanism" == "temporal_pos" ] ; then
#     wget https://www.dropbox.com/s/tae3rdm9ouaicfb/hmnist_temporal_pos.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
# fi