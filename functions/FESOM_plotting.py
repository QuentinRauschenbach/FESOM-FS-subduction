import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import cmocean.cm as cmo
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def simple_map(x2: np.ndarray, y2: np.ndarray, elements: np.ndarray, data: np.ndarray, levels: list, 
               projection=ccrs.NorthPolarStereo(), extent: list = [-20, 20, 75, 82], 
               title: str = None, cbar_label: str = None, cmap=cm.thermal, 
               ticks: list = None, figsize: tuple = (8, 5), cbar_extent: str = None) -> tuple:
    """
    Create a simple map with contour fill using Cartopy and Matplotlib.

    Parameters:
    -----------
    x2 : np.ndarray
        Array of x-coordinates (e.g., longitude) for the data points.
    y2 : np.ndarray
        Array of y-coordinates (e.g., latitude) for the data points.
    elements : np.ndarray
        Array defining the triangular elements for plotting (e.g., mesh connectivity).
    data : np.ndarray
        Array of data values to be plotted (e.g., temperature, salinity).
    levels : list
        List of contour levels for the plot.
    projection : cartopy.crs.Projection, optional
        Cartopy projection for the map. Default is `ccrs.NorthPolarStereo()`.
    extent : list, optional
        List defining the map extent in the format [lon_min, lon_max, lat_min, lat_max].
        Default is `[-20, 20, 75, 82]`.
    title : str, optional
        Title of the plot. Default is `None`.
    cbar_label : str, optional
        Label for the colorbar. Default is `None`.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the plot. Default is `cmo.cm.thermal`.
    ticks : list, optional
        List of tick values for the colorbar. Default is `None`.
    figsize : tuple, optional
        Tuple defining the figure size (width, height) in inches. Default is `(8, 5)`.
    cbar_extent : str, optional
        Colorbar extension (e.g., "both", "min", "max"). Default is `None`.

    Returns:
    --------
    tuple
        A tuple containing the Matplotlib figure and axes objects (`fig`, `ax`).
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection}, constrained_layout=True)
    
    # Set map extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add features
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    
    # Plot data
    im = ax.tricontourf(x2, y2, elements, data, cmap=cmap, extend=cbar_extent, levels=levels)
    
    # Colorbar
    cbar = fig.colorbar(im)
    cbar.set_label(cbar_label)
    cbar.set_ticks(ticks)
    
    # Title
    plt.title(title, fontweight='bold', fontsize=14)
    
    return fig, ax