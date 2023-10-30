import torch

def logmeanexp(x, dim, eps=1e-5):
    """
        Log-mean-exp.
        : args x        : Reducing tensor.
        : args dim      : The dimensions to reduce.
    """
    x_max = torch.max(x, dim, keepdim=True).values
    return torch.log(torch.mean(torch.exp(x-x_max), dim=dim, keepdim=True) + eps) + x_max

def gen_mask_t(t, time_length):
        return torch.Tensor([(i== ((t + time_length) % time_length)) for i in range(time_length)]).bool().unsqueeze(0).unsqueeze(-1)

def moving_avg(loss, nll, kl, smi, mae, mse, rmse, mre, mnll, curr_loss, curr_nll, curr_kl, curr_smi, curr_mae, curr_mse, curr_rmse, curr_mre, curr_mnll, num_samples, batch_size, num_missing, curr_missing):
    curr_loss = curr_loss * (num_samples - batch_size) / num_samples + loss.detach().item() * batch_size / num_samples
    curr_nll  = curr_nll  * (num_samples - batch_size) / num_samples + nll.detach().item()  * batch_size / num_samples
    curr_kl   = curr_kl   * (num_samples - batch_size) / num_samples + kl.detach().item()   * batch_size / num_samples
    curr_smi  = curr_smi  * (num_samples - batch_size) / num_samples + smi.detach().item()  * batch_size / num_samples
    curr_mae  = curr_mse  * (num_missing - curr_missing) / num_missing + mae.detach().item()  * curr_missing / num_missing
    curr_mse  = curr_mse  * (num_missing - curr_missing) / num_missing + mse.detach().item()  * curr_missing / num_missing
    curr_rmse = curr_mse  * (num_missing - curr_missing) / num_missing + rmse.detach().item() * curr_missing / num_missing
    curr_mre  = curr_mse  * (num_missing - curr_missing) / num_missing + mre.detach().item()  * curr_missing / num_missing
    curr_mnll = curr_mnll * (num_missing - curr_missing) / num_missing + mnll.detach().item() * curr_missing / num_missing
    return curr_loss, curr_nll, curr_kl, curr_smi, curr_mae, curr_mse, curr_rmse, curr_mre, curr_mnll