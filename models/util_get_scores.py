import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

class predictive_scorer(nn.Module):
    def __init__(self, args):
        super(predictive_scorer, self).__init__()
        self.input_dim = args.data_dim
        self.hidden_dim = self.input_dim // 2
        self.net_rnn = nn.LSTM(input_size = self.input_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)
        self.net_nn = nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim)

    def forward(self, x):
        # x: [B, T-1, D]
        h, _ = self.net_rnn(x)
        x_hat = self.net_nn(h) # x_hat: [B, T-1, D]
        return x_hat 

def get_imputed_inputs(model, loader, device, is_label, test):
    full, imputed, ys = [], [], []
    # print("Getting imputed inputs..")
    # for idx, batch in enumerate(tqdm(loader)):
    for idx, batch in enumerate(loader):
        x_full, x_miss, m_miss, _, y, t= batch
        x_miss, m_miss, t = x_miss.to(device),  m_miss.to(device), t.to(device)
        x_imputed = model.sample_classify(x_miss, t, m_miss).detach().cpu()
        if test: x_imputed = x_miss.detach().cpu()
        x_imputed = torch.where(m_miss.bool().cpu(), x_imputed, x_miss.cpu()) # Impute only missing features.
        full.append(x_full)
        imputed.append(x_imputed)
        ys.append(y)
    if is_label: return np.concatenate(full), np.concatenate(imputed), np.concatenate(ys)
    else: return np.concatenate(full), np.concatenate(imputed), None

def get_predictive_score(x_full, x_imputed, device, args):
    """
        Reconstruct the next timestep given remaining timesteps.
    """
    max_iter = 50000
    x_full, x_imputed = torch.from_numpy(x_full).to(device), torch.from_numpy(x_imputed).to(device)
    model = predictive_scorer(args).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    val_split = len(x_imputed) // 2
    
    # Train the predictor
    loss_func = nn.L1Loss()
    for iter in tqdm(range(max_iter)):
        x_predict = model(x_imputed[:val_split, :-1])
        loss = loss_func(x_predict, x_full[:val_split, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if iter % 1000 == 0: print(f"iter {iter}: {loss}")
    # Test the predictor
    x_predicted = model(x_imputed[val_split:, :-1])
    mae = loss_func(x_predicted, x_full[val_split:, 1:]).detach().cpu()
    return mae

def get_discriminative_score(x_imputed, y, args):
    """
        Classifiy the timestep or the timeseries.
    """
    if args.dataset in ["hmnist", "rotated"]: 
        y_round = y
        if args.dataset in ["hmnist", "rotated"]: x_imputed_round = np.round(x_imputed)
        else: x_imputed_round = x_imputed
        x_imputed_round = x_imputed_round.reshape([-1, args.time_length * args.data_dim])
        val_split = len(x_imputed_round) // 2
        classifier_round = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-10, max_iter=10000)
        classifier_round.fit(x_imputed_round[:val_split], y_round[:val_split])
        probs_round = classifier_round.predict_proba(x_imputed_round[val_split:])
        auprc_round = average_precision_score(np.eye(args.num_classes)[y_round[val_split:]], probs_round)
        auroc_round = roc_auc_score(np.eye(args.num_classes)[y_round[val_split:]], probs_round)
        
        return auroc_round, auprc_round
    
    elif args.dataset in ["physionet"] :
        x_imputed = x_imputed.reshape([-1, args.time_length * args.data_dim])
        val_split = len(x_imputed) // 2
        classifier = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000)
        y = y.ravel()
        classifier.fit(x_imputed[:val_split], y[:val_split])
        probs = classifier.predict_proba(x_imputed[val_split:])[:, 1]
        auprc = average_precision_score(y[val_split:], probs)
        auroc = roc_auc_score(y[val_split:], probs)
        
        return auroc, auprc