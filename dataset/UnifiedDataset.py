import numpy as np
import torch

is_artificial = ["physionet"]
is_label = ["hmnist", "physionet", "rotated", "adni"]

class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, train:bool, test:bool, args):
        data = np.load(args.datadir, allow_pickle=True)
        self.dataset = args.dataset
        print([i for i in data.keys()])
        if not test: ## Train or Valid
            if train: ## Train
                x_full = data['x_train_full']
                x_miss = data['x_train_miss']
                m_miss = data['m_train_miss']
                if 'm_train_artificial' in data.keys(): m_artificial = data["m_train_artificial"]
                y = data['y_train']
                t = data['t_train']
                
            else: ## Valid
                x_full = data["x_valid_full"]  # full for artificial missings
                x_miss = data["x_valid_miss"]
                m_miss = data["m_valid_miss"]
                if 'm_valid_artificial' in data.keys(): m_artificial = data["m_valid_artificial"]
                y = data["y_valid"]
                t = data['t_valid']
                
        else: ## Test
            x_full = data['x_test_full']
            x_miss = data['x_test_miss']
            m_miss = data['m_test_miss']
            if 'm_test_artificial' in data.keys(): m_artificial = data["m_test_artificial"]
            y = data['y_test']
            t = data['t_test']
        
        self.x_full = x_full
        self.x_miss = x_miss
        self.m_miss = m_miss
        self.m_artificial = m_miss
        if self.dataset in is_artificial: self.m_artificial = m_artificial
        if self.dataset in is_label: self.is_label=True
        else: self.is_label=False
        self.y = y
        self.t = t
        
        collate = Collate(args.imputed, args.time_length, args.dataset, self.is_label) # function for generating a minibatch from strings
        self.loader = torch.utils.data.DataLoader(self, collate_fn=collate, batch_size=args.batch_size, shuffle=False) # num_workers=1, 

    def __len__(self):
        return len(self.x_full)

    def __getitem__(self, idx):
        if self.is_label: return (self.x_full[idx], self.x_miss[idx], self.m_miss[idx], self.m_artificial[idx], self.y[idx], self.t[idx])
        else: return (self.x_full[idx], self.x_miss[idx], self.m_miss[idx], self.m_artificial[idx], None, self.t[idx])

class Collate:
    def __init__(self, is_imputed, time_length, dataset, is_label):
        self.is_imputed = is_imputed
        self.time_length = time_length
        self.dataset = dataset
        self.is_label = is_label
        pass

    def __call__(self, batch):
        """
        Returns a minibatch of images.
        """
        batch_size = len(batch)
        x_full, x_miss, m_miss, m_artificial, y, t = [], [], [], [], [], []
    
        for index in range(batch_size):
            x_full.append(batch[index][0])
            x_miss.append(batch[index][1])
            m_miss.append(batch[index][2])
            m_artificial.append(batch[index][3])
            if self.is_label: y.append(batch[index][4])
            t.append(batch[index][5])

        x_full = torch.Tensor(np.array(x_full))
        x_miss = torch.Tensor(np.array(x_miss))
        m_miss = torch.Tensor(np.array(m_miss))
        m_artificial = torch.Tensor(np.array(m_artificial))
        if self.is_label: y = torch.Tensor(np.array(y)).long()
        else: y = None
        t = torch.Tensor(np.array(t))
        
        if self.is_imputed == "forward": x_miss = self.forward_imputation(x_miss, m_miss)
        elif self.is_imputed == "mean": x_miss = self.mean_imputation(x_miss, m_miss)

        return (x_full, x_miss, m_miss, m_artificial, y, t)
    
    def mean_imputation(self, x_miss, m_miss):
        x_mean = torch.sum(x_miss, dim=-2, keepdim=True) / torch.sum((~(m_miss.bool())).float(), dim=-2, keepdim=True)
        x_mean = torch.nan_to_num(x_mean)
        x_mean = torch.tile(x_mean, (1, self.time_length, 1))
        m_miss = m_miss.bool()
        x_imputed = torch.where(~m_miss, x_miss, x_mean)

        return x_imputed

    def forward_imputation(self, x_miss, m_miss):
        m_miss = m_miss.bool()
        x_fwd = x_miss.clone()
        x_bwd = x_miss.clone()
        x_is_observed_fwd = ~m_miss.clone().bool()
        x_is_observed_bwd = ~m_miss.clone().bool()

        if self.dataset == "hmnist":
            for t in range(self.time_length-1):
                x_fwd[:,t+1] = torch.where(~m_miss[:,t+1], x_miss[:,t+1], x_fwd[:,t]).bool().float()
                x_is_observed_fwd[:,t+1] = (x_is_observed_fwd[:,t+1] + x_is_observed_fwd[:,t]).bool()
                x_bwd[:,-2-t] = torch.where(~m_miss[:,-2-t], x_miss[:, -2-t], x_bwd[:, -1-t]).bool().float()
                x_is_observed_bwd[:,-2-t] = (x_is_observed_bwd[:,-2-t] + x_is_observed_bwd[:,-1-t]).bool()
            x_imputed = torch.where(~m_miss, x_miss, torch.where(x_is_observed_fwd, x_fwd, torch.where(x_is_observed_bwd, x_bwd, x_miss)))
        else:
            for t in range(self.time_length-1):
                x_fwd[:,t+1] = torch.where(~m_miss[:,t+1], x_miss[:,t+1], x_fwd[:,t]).float()
                x_is_observed_fwd[:,t+1] = (x_is_observed_fwd[:,t+1] + x_is_observed_fwd[:,t]).bool()
                x_bwd[:,-2-t] = torch.where(~m_miss[:,-2-t], x_miss[:, -2-t], x_bwd[:, -1-t]).float()
                x_is_observed_bwd[:,-2-t] = (x_is_observed_bwd[:,-2-t] + x_is_observed_bwd[:,-1-t]).bool()
            x_imputed = torch.where(~m_miss, x_miss, torch.where(x_is_observed_fwd, x_fwd, torch.where(x_is_observed_bwd, x_bwd, x_miss)))

        return x_imputed