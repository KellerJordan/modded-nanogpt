import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from lib.geoopt.manifolds.lorentz.math import lorentz_to_poincare, poincare_to_lorentz

import umap

#mpl.rcParams.update({
#    'text.usetex' : True ,
#    "font.family": "serif",
#    'font.size': 20
#})

@torch.no_grad()
def visualize_reconstructions(model, dataloader, device, num_imgs: int = 5):
    """ Visualizes image reconstructions of a VAE-model. 
    
    Dataloader has to have a batch_size > num_imgs!

    Returns a matplotlib.pyplot figure.
    """
    model.eval()
    model.to(device)

    x, _ = next(iter(dataloader))

    x = x[:num_imgs] # Select first images
    x = x.to(device)
    x_hat = model.module.reconstruct(x)

    x = x.cpu().detach().numpy()
    x_hat = x_hat.cpu().detach().numpy()

    fig = plt.figure()

    for i in range(num_imgs):
        # Plot input img
        ax = fig.add_subplot(2, num_imgs,i+1, xticks=[], yticks=[])
        plt.imshow(x[i].transpose(1,2,0), cmap='gray')

        # Plot reconstructed img
        ax = fig.add_subplot(2, num_imgs,(i+1)+num_imgs, xticks=[], yticks=[])
        plt.imshow(x_hat[i].transpose(1,2,0), cmap='gray')
        
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

@torch.no_grad()
def visualize_generations(model, device, num_imgs_per_axis: int = 5):
    """ Visualizes image generations of a VAE-model. 

    Returns a matplotlib.pyplot figure.
    """
    model.eval()
    model.to(device)

    x_gen = model.module.generate_random(num_imgs_per_axis**2, device)
    x_gen = x_gen.cpu().detach().numpy()

    fig = plt.figure(figsize=(10,10))
    for i in range(num_imgs_per_axis**2):
        # Plot input img
        ax = fig.add_subplot(num_imgs_per_axis, num_imgs_per_axis, i+1, xticks=[], yticks=[])
        plt.imshow(x_gen[i].transpose(1,2,0), cmap='gray') 

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
        
    return fig

@torch.no_grad()
def visualize_hyperbolic(data, device, manifold = None, poincare=False, labels=None):
    """ Plots hyperbolic data on Poincaré ball and tangent space 

    Note: This function only supports curvature k=1.
    """
    if labels is not None:
        labels = labels.cpu().numpy()

    fig = plt.figure(figsize=(14,7))

    # 2D embeddings
    if (data.shape[-1]==2 and poincare) or (data.shape[-1]==3 and not poincare):
        if poincare:
            data_P = data.cpu()
        else:
            data_P = lorentz_to_poincare(data, k=manifold.k).cpu()
    # Dimensionality reduction to 2D
    else:
        if poincare:
            data = poincare_to_lorentz(data, manifold.k)
        reducer = umap.UMAP(output_metric='hyperboloid')
        data = reducer.fit_transform(data.cpu().numpy())
        data = manifold.add_time(torch.tensor(data).to(device))
        data_P = lorentz_to_poincare(data, k=manifold.k).cpu()

    ax = fig.add_subplot(1,2,1)
    plt.scatter(data_P[:,0], data_P[:,1], c=labels, s=1)
    # Draw Poincaré boundary
    boundary=plt.Circle((0,0),1, color='k', fill=False)
    ax.add_patch(boundary)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal', adjustable='box')

    plt.colorbar()
    plt.xlabel("$z_0$")
    plt.ylabel("$z_1$")
    ax.set_title("Poincaré Ball")

    # Plot hyperbolic embeddings in tangent space of the origin
    if poincare:
        z_all_T = (manifold.logmap0(data_P.to(device))).detach().cpu()
    else:
        z_all_T = (manifold.logmap0(data)).detach().cpu()
        z_all_T = z_all_T[..., 1:]

    ax = fig.add_subplot(1,2,2)
    plt.scatter(z_all_T[:,0], z_all_T[:,1], c=labels, s=1)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.xlabel("$z_0$")
    plt.ylabel("$z_1$")
    ax.set_title("Tangent Space")

    return fig

@torch.no_grad()
def visualize_embeddings(model, dataloader, device, manifold = None, poincare=False):
    """ Visualizes embeddings of a model. 

    Umap only supports k=1?
    """
    model.eval()

    z_all = []
    labels = []

    model.to(device)

    for x, y in dataloader:
        x = x.to(device)
        z = model.module.embed(x)

        z_all.extend(z.cpu().detach().numpy().tolist())
        labels.extend(y.numpy().tolist())

    z_all = torch.tensor(z_all, device=device) # gpu or cpu
    labels = torch.tensor(labels) # cpu

    if manifold is not None:
        fig = visualize_hyperbolic(z_all, device, manifold, poincare, labels)

    else:
        # Plot Euclidean embeddings
        if z_all.shape[-1]>2:
            reducer = umap.UMAP()
            z_all = reducer.fit_transform(z_all.cpu().numpy())
        else:
            z_all = z_all.detach().cpu()
        
        fig = plt.figure(figsize=(14, 7))

        ax = fig.add_subplot(1,2,1)
        plt.scatter(z_all[:,0], z_all[:,1], c=labels, s=1)
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")
        
    return fig


