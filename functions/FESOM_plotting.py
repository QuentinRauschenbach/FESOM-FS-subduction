import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from typing import Optional, Sequence, Tuple, Union

def plot_polar_contourf(
    mesh: object,
    data: np.ndarray,
    *,
    levels: Optional[Union[np.ndarray, Sequence[float]]] = None,
    cmap: str = 'viridis',
    extend: str = 'both',
    flip_cbar: bool = False,
    extent: Sequence[float] = [-20, 20, 75, 82],
    label: str = "Value",
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    fill_num: Optional[float] = -100,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot a tricontourf map of unstructured data on a polar projection.

    Parameters
    ----------
    mesh : object
        Mesh object with attributes:
        - `x2`: 1D numpy array of longitudes (or x-coordinates).
        - `y2`: 1D numpy array of latitudes (or y-coordinates).
        - `elem`: 2D array-like of triangle indices for the mesh.
    data : np.ndarray
        1D array of scalar values defined on the mesh.
    dx : float, optional
        Interval between contour levels (used if `levels` is None), by default 100.
    levels : array-like, optional
        Contour levels. If None, levels are automatically generated from data using `dx`.
    cmap : str, optional
        Name of the matplotlib colormap to use, by default 'viridis'.
    projection : cartopy.crs.Projection, optional
        Map projection for the plot, by default `ccrs.NorthPolarStereo()`.
    extent : list of float, optional
        Geographic extent in [lon_min, lon_max, lat_min, lat_max] (degrees), by default [-20, 20, 75, 82].
    label : str, optional
        Label for the colorbar, by default "Value".
    figsize : tuple of float, optional
        Size of the figure in inches (width, height), by default (8, 6).
    title : str, optional
        Title of the plot, by default None.
    show : bool, optional
        Whether to display the plot using `plt.show()`, by default True.
    save_path : str, optional
        If provided, saves the figure to this path (e.g., "output.png"), by default None.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        If `show` is False, returns the figure and axis for further use. Otherwise, returns None.
    """
    projection = ccrs.NorthPolarStereo()
    pc = ccrs.PlateCarree()
    x_proj, y_proj = projection.transform_points(pc, mesh.x2, mesh.y2)[:, :2].T

    if levels is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        levels = np.linspace(vmin, vmax+10, 10)

    # Check if the array contains NaNs
    if np.any(np.isnan(data)):
        print(f"Replace NaNs with {fill_num}")
        data = np.nan_to_num(data, nan=fill_num)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection}, constrained_layout=True)
    cf = ax.tricontourf(x_proj, y_proj, mesh.elem, data, levels=levels, cmap=cmap, extend=extend)

    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(label)
    if flip_cbar:
        cbar.ax.invert_yaxis()  # Flips the colorbar so deeper values are at the bottom


    ax.set_extent(extent, crs=pc)
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, 
                  color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = False   # Show right y-axis labels
    

    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
        return None
    else:
        return fig, ax