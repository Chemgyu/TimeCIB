import torch

def rbf_kernel(T, length_scale):
    xs = torch.arange(T)
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out)**2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the "\
                               "kernel matrix to be diagonally dominant"
    sigmas = torch.ones(T, T) * length_scale
    sigmas_tridiag = torch.tril(sigmas) + torch.triu(sigmas) - torch.trace(sigmas)
    kernel_matrix = sigmas_tridiag + torch.eye(T)*(1. - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = torch.arange(T)
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = torch.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / torch.sqrt(length_scale)
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(T)
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out)**2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = torch.divide(sigma, (distance_matrix_scaled + 1.))

    alpha = 0.001
    eye = torch.eye(list(kernel_matrix.size())[-1])
    return kernel_matrix + alpha * eye
