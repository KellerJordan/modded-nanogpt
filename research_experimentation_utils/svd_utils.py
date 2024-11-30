import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

import torch

@torch.compile
def annotated_zeropower_via_newtonschulz5(G, steps=10, eps=1e-12):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    # a, b, c = (3.4445, -4.7750,  2.0315)
    # a, b, c = (3.013, -3.601, 1.523)
    a, b, c = (4, -4.8, 1.5)
    X = G
    X = X / (X.norm() + eps) * 1.2 # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    singular_values_over_iters = []
    for i in range(steps):
        S = torch.linalg.svdvals(X)
        singular_values_over_iters.append(S.tolist())
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X, singular_values_over_iters

def get_singular_values(matrix_params, newton_schulz_iters):
    list_of_singular_value_lists_by_matrix = []
    for params in matrix_params:
        _, singular_value_list = annotated_zeropower_via_newtonschulz5(params, steps=newton_schulz_iters)
        list_of_singular_value_lists_by_matrix.append(singular_value_list)

    records_of_singular_value_lists_by_ns_iter = []
    for newton_schulz_iter in range(newton_schulz_iters):
        singular_values_for_this_iter = []
        for single_matrix_singular_values_list in list_of_singular_value_lists_by_matrix:
            singular_values_for_this_iter += single_matrix_singular_values_list[newton_schulz_iter]
        singular_values_for_this_iter.sort()
        records_of_singular_value_lists_by_ns_iter.append({
            "newton_schulz_iter": newton_schulz_iter,
            "singular_values": singular_values_for_this_iter
        })
    return pd.DataFrame(records_of_singular_value_lists_by_ns_iter)

def animate_singular_values(df, interval=200, bins=30):
    frames = []
    
    all_values = np.concatenate(df['singular_values'].values)
    value_range = (float(np.min(all_values)), float(np.max(all_values)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(i):
        ax.clear()
        current_data = np.concatenate(df[df['newton_schulz_iter'] == i]['singular_values'].values)
        ax.hist(current_data, bins=bins, 
               range=value_range,
               alpha=0.7, color='blue')
        ax.set(xlabel='Singular Values', 
               ylabel='Frequency',
               title=f'Singular Values at Newton-Schulz Iteration {i}')
        ax.grid(alpha=0.3)
        
    anim = FuncAnimation(
        fig, 
        animate,
        frames=int(df['newton_schulz_iter'].max()) + 1,
        interval=interval
    )
    
    # Save as MP4 using PillowWriter
    temp_gif = "singular_values.gif"
    anim.save(temp_gif, writer=PillowWriter(fps=1000/interval))
    plt.close()
    
    return temp_gif