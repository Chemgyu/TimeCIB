import torch

def cauchy_contrast(T, sigma, length_scale):
    xs = torch.arange(T)
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out)**2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = torch.divide(sigma**2, (distance_matrix_scaled + 1.))

    return kernel_matrix

def sine_contrast(T, sigma, length_scale, period_scale):
    xs = torch.arange(T)
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out)
    distance_matrix_sin = torch.pi * distance_matrix / period_scale
    kernel_matrix = (sigma**2) * torch.exp(- torch.divide( 2 * torch.sin(distance_matrix_sin),  (length_scale ** 2)))

    return kernel_matrix

def cauchy_contrast_max(T, sigma, length_scale):
    dist_max = (T - 0)**2
    dist_scaled = dist_max / length_scale ** 2
    kernel_coef = torch.divide(sigma, (dist_scaled + 1.))
    return kernel_coef

def uniform_contrast(T):
    kernel_matrix = torch.ones(T, T)
    return kernel_matrix