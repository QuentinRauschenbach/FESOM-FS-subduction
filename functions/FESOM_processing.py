import numpy as np
import xarray as xr

def reshape_3d_nodes(ds: xr.Dataset, var: str, time_idx: int, fill_value: int = -999) -> np.ndarray:
    """
    Reshapes 3D node data from a FESOM dataset into a (2D node, depth) format.
    The dataset must have `aux3d` data with shape `(num_nodes_2d, num_depth_levels)` where each entry
    in `aux3d` corresponds to the 3D node number for each 2D node and depth level.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the 3D ocean model data.
    var : str
        The variable name (e.g., 'temp') to reshape.
    time_idx : int
        The time index for extracting data.
    fill_value : int, optional (default = -999)
        The value representing missing data in `aux3d_mapping`.

    Returns:
    --------
    np.ndarray
        A (num_nodes_2d, num_depth_levels) shaped array with the reshaped values.
    """

    # Read aux3d (Mapping wich 3D nodes are above each other)
    aux3d_mapping = ds['aux3d'].astype(int).values
    
    # Get data for one time step
    data_raw = ds[var].isel(time=time_idx)
    
    # Get dimension size & initialize reshaped array
    num_nodes_2d     = ds['nodes_2d'].size
    num_depth_levels = ds['depth_levels'].size
    data_reshaped = np.full((num_nodes_2d, num_depth_levels), np.nan)
    
    # Find the valid 3D indices for all nodes (where aux is actually refering to an existing node)
    valid_mask = (aux3d_mapping != fill_value) & (aux3d_mapping >= 0)
    valid_aux3d_indices = aux3d_mapping[valid_mask] # results in flat 1D boolean vector of valid entry positions
                                                    # this is not in the 0, 1, ...  num_3d_nodes order but goes trough each depth collumn
    
    # Get the corresponding (i, j) indices for the reshaped array
    row_indices_2d, col_indices_depth = np.where(valid_mask) # returns the row and column indices of valid entry positions
    
    # Perform the vectorized indexing
    # resort values
    original_values = data_raw.isel(nodes_3d=valid_aux3d_indices).values # resorts the values
    
    # Assign these values to the correct positions in the reshaped array
    data_reshaped[row_indices_2d, col_indices_depth] = np.where(np.isnan(original_values), np.nan, original_values)

    return data_reshaped