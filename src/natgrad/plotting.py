from matplotlib import pyplot as plt
from natgrad.domains import Hyperrectangle, Domain
from jax import numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def default_axis(func):
    def new_func(*args, **kwargs):
        return func(*args, ax=kwargs.pop("ax", plt.gca()), **kwargs)
    return new_func
    

def plot_2d_func(func, domain, N=200, ax: plt.Axes = None, **kwargs):
    """Plot a 2d domain.

    Args:
        domain (tuple): A tuple of two floats (a, b) representing the domain of a function.
    """
    bounding_box: Hyperrectangle = domain.bounding_box()
    lb, rb = bounding_box._l_bounds, bounding_box._r_bounds
    X, Y = jnp.meshgrid(jnp.linspace(lb[0], rb[0], N), jnp.linspace(lb[1], rb[1], N))
    xyflat = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    Z = func(xyflat).reshape((N, N))
    mask = vmap(domain.mask, (0))(xyflat).reshape((N, N))
    Z = jnp.where(mask, Z, jnp.nan)
    return ax.pcolormesh(X, Y, Z, **kwargs)


def plot_2d_funcs(funcs, domain, axList: List[plt.Axes], N=200, same_vlim=False, **kwargs):
    """Plot a 2d domain.

    Args:
        domain (tuple): A tuple of two floats (a, b) representing the domain of a function.
    """
    bounding_box: Hyperrectangle = domain.bounding_box()
    lb, rb = bounding_box._l_bounds, bounding_box._r_bounds
    X, Y = jnp.meshgrid(jnp.linspace(lb[0], rb[0], N), jnp.linspace(lb[1], rb[1], N))
    xyflat = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    mask = vmap(domain.mask, (0))(xyflat).reshape((N, N))
    
    Zlist = []
    vmin, vmax = jnp.inf, -jnp.inf
    for func in funcs:
        # Evaluate function, mask out excluded regions
        Zlist.append(jnp.where(mask, func(xyflat).reshape((N, N)), jnp.nan))
        vmin, vmax = min(vmin, jnp.nanmin(Zlist[-1])), max(vmax, jnp.nanmax(Zlist[-1]))
    vmin, vmax = (vmin, vmax) if same_vlim else (None, None)
    return [ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, **kwargs) for ax, Z in zip(axList, Zlist)]


def remove_axes(ax):
    """
    Remove axes from plot.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return None
    
    
# function to remove top and right axes from plot
def remove_top_right_axes(ax):
    """
    Remove top and right axes from plot.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return None


# function to plot a horizontal bar chart with specified labels and values  
def barplot(ax, labels, values, title=None, xlabel=None, ylabel=None):
    """
    Plot a horizontal bar chart.
    """
    # remove top and right axes
    remove_top_right_axes(ax)
    # plot the bars
    ax.barh(labels, values)
    # set y ticks labels to be the labels
    #ax.set_yticks(labels)
    # set the title
    if title is not None:
        ax.set_title(title)
    # set the x and y labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return None 


def pretty_pyplot_layout():
    # Change matplotlib settings to be more presentable
    plt.rcParams['figure.figsize'] = [10, 5]
    #plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 14
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'medium'
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['axes.labelsize'] = 'medium'
    plt.rcParams['xtick.labelsize'] = 'medium'
    plt.rcParams['ytick.labelsize'] = 'medium'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    #plt.rcParams['grid.color'] = "#cccccc"
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['figure.autolayout'] = True
    # pretty sans-serif fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
    plt.rcParams['text.usetex'] = True



# Helper function to plot a function with periodic boundary conditions
def plot_periodic(xmin, xmax, pad, func, N, **plotargs):
    x = np.linspace(xmin-pad, xmax+pad, N)
    x_wrap = (x - xmin) % (xmax-xmin) + xmin
    left_mask = x < xmin
    right_mask = x > xmax
    inner = np.logical_not(np.logical_or(left_mask, right_mask))
        
    # plot function values and mark wrapped values with a different color
    leg = plotargs.pop('label', None)
    plt.plot(x[inner], func(x_wrap[inner]), label=leg, **plotargs)
    plt.plot(x[left_mask], func(x_wrap[left_mask]), alpha=0.5, **plotargs)
    plt.plot(x[right_mask], func(x_wrap[right_mask]), alpha=0.5, **plotargs)


# Function to plot a function evaluated on a line 
def plot_line(xval, yval, funcs, titles, domain_plotter=None, axis='x'):

    plt.figure(figsize=(20,5))

    if axis=='x':
        xplot = xval
    else:
        xplot = yval

    # Solution
    plt.subplot(131)
    plt.title("Solution on line")
    func_vals = [f(xval, yval) for f in funcs]
    for title, val in zip(titles, func_vals):
        plt.plot(xplot, val, label=title)
    remove_top_right_axes(plt.gca())
    plt.legend()

    # Errors
    plt.subplot(132)
    plt.title("Error on line")
    for title, val in zip(titles[1:], func_vals[1:]):
        plt.plot(xplot, val - func_vals[0], label=title)
    remove_top_right_axes(plt.gca())
    plt.legend()

    # Domain
    plt.subplot(133)
    plt.title("Evaluation Setup")
    if domain_plotter is not None:
        domain_plotter(plt.gca())
    plt.plot(xval, yval, 'r--', linewidth=2)

# Function to plot quantiles of time series data over a time window
def plot_quantiles(ax, x: np.ndarray, y: np.ndarray, quantiles: np.ndarray, window: int, **kwargs):
    
    y_qs = np.zeros((len(y), len(quantiles)))
    y_mean = np.zeros(len(y))

    for i in range(len(x)):
        # Make window (avoid invalid idx errors)
        i_low = max(0, i - window//2)
        i_hig = min(len(x), i + window//2)

        # Take mean and find quantiles
        y_vals = y[i_low:i_hig]
        y_mean[i] = np.mean(y_vals)
        y_qs[i, :] = np.quantile(y_vals, quantiles)
    
    label = kwargs.pop("label", "")
    for j in range(len(quantiles)):
        ax.fill_between(x, y_mean, y_qs[:, j], alpha=0.5/len(quantiles), **kwargs)
    
    ax.plot(x, y_mean, label=label, **kwargs)

