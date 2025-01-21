import numpy as np
from prettytable import PrettyTable
from scipy.integrate import quad
import torch
import math 
def cv_monte_carlo_integration(f, cv_fn, g_integral, domain, n_samples, device = torch.device('mps'), chunk_size=1000, repeat = True):
    """
    Monte Carlo integration with control variates.

    Args:
        f (function): The target function to integrate.
        g (function): Control variate function, closely resembling f.
        g_integral (float): Analytical integral of the control variate g over the domain.
        domain (tuple): The integration domain (start, end).
        n_samples (int): Number of samples.

    Returns:
        float: Estimated integral of f.
    """
    start, end = domain
    integral_estimates = []
    residuals_list = []
    if repeat:
        runs = 5
    else:
        runs = 1
    for _ in range(runs):
        # Generate random samples in the domain
        samples = torch.rand(n_samples) * (end - start) + start
        # Compute the function values
        f_values = f(samples)
        g_values_list = []
        with torch.no_grad():
            # Split the 'samples' tensor into smaller chunks
            for i in range(0, n_samples, chunk_size):
                batch = samples[i : i + chunk_size]
                # Pass shape [chunk_size, 1] to the network
                batch = batch.view(-1, 1).to(device)
                g_output = cv_fn.compute_graph_fast2({'x_coords': batch, 'params': None})
                g_values_list.append(g_output.detach().cpu())
        g_values = torch.cat(g_values_list, dim=0)
        g_values = g_values.detach().cpu()
        residual = f_values - g_values.squeeze()
        # Monte Carlo estimate with control variates
        integral_estimate = g_integral + (end - start) * torch.mean(residual)
        integral_estimates.append(integral_estimate.item())
        residuals_list.append(residual)

    
    if repeat:
        integral_estimates = torch.tensor(integral_estimates)
        mean_integral_estimate = torch.mean(integral_estimates)
        std_integral_estimate = torch.std(integral_estimates)
        return mean_integral_estimate, std_integral_estimate
    else:
        integral_estimates = torch.tensor(integral_estimates)
        mean_integral_estimate = torch.mean(integral_estimates)
        return integral_estimate

def display_integration_results(func, domain, cv_fn, g_integral, sample_sizes, device):
    # Numerical quadrature integration
    quad_result, _ = quad(func, domain[0], domain[1])

    # Create a PrettyTable
    table = PrettyTable()
    table.field_names = ["Sample Size", "Integration Result (mean ± std.dev)"]

    # Compute and add results for different sample sizes
    for n_samples in sample_sizes:
        mc_mean, mc_std = cv_monte_carlo_integration(func,cv_fn,g_integral, domain, n_samples, device)
        table.add_row([n_samples, f"{mc_mean.item():.4f} ± {mc_std.item():.4f}"])

    # Add the quadrature result to the table
    table.add_row(["Scipy Quad", f"{quad_result:.4f} ± 0.0000"])

    # Print the table
    print(table)
    
def monte_carlo_integration(func, a, b, num_samples):
    samples = np.random.uniform(a, b, num_samples)
    func_values = func(samples)
    average_value = np.mean(func_values)
    integral_estimate = (b - a) * average_value
    return integral_estimate