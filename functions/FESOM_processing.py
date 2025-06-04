import numpy as np
import xarray as xr
import pandas as pd

def load_FESOM_data_with_grid(meshpath:str, datapath:str, vars:str="oce") -> xr.Dataset:
    """
    Load FESOM data and grid information from the specified paths.
    Parameters:
    ---------- 
    meshpath : str
        Path to the mesh files.
    datapath : str
        Path to the data files.
    vars : str
        Variable type to load (default is "oce").
    Returns:
    -------
    ds : xarray.Dataset
        Dataset containing the loaded data and grid information.
    """
    
    ds   = xr.open_dataset(datapath)
    
    if vars == "oce":
        df_bath  = pd.read_csv(meshpath + "depth.out", skiprows=0, names=["bath"])
        df_depth = pd.read_csv(meshpath + "nod3d.out", skiprows=1, names=["knot_num", "lon", "lat", "depth", "border_id"], sep='\s+', index_col=0)
        df_aux3d = pd.read_csv(meshpath + "aux3d.out", skiprows=1, names=["knot_num"])

        data = xr.DataArray(df_bath['bath'].values, dims=['nodes_2d'])
        data.attrs['description'] = 'depth from depth.out'
        data.attrs['units'] = 'meters'
        ds['bath'] = data

        data = xr.DataArray(df_depth['depth'].values, dims=['nodes_3d'])
        data.attrs['description'] = 'depth of each 3D node from nod3d.out'
        data.attrs['units'] = 'meters'
        ds['depth'] = data

        depth_levels = np.flip(np.array(sorted(list(set(ds.depth.values))))) #np.unique
        depth_levels = np.append(depth_levels, [np.nan, np.nan]) # fill the last two with nan (not present in Fram Strait data)

        # Add depth_levels as a coordinate to ds
        ds = ds.assign_coords(depth_levels=depth_levels)
        ds.depth_levels.attrs['description'] = 'model depth levels'
        ds.depth_levels.attrs['units'] = 'meters'

        end = 570732*47
        data = xr.DataArray(df_aux3d['knot_num'].values[:end].reshape(-1,47)-1, dims=['nodes_2d', "depth_levels"])
        ds['aux3d'] = data
        ds['aux3d'].attrs['description'] = 'Mapping of 3D nodes to 2D nodes and depth from aux3d.out'
        ds['aux3d'].attrs['units'] = '1'
        ds['aux3d'].attrs['missing_value'] = -999
    
    df_lonlat_2d = pd.read_csv(meshpath + "nod2d.out", skiprows=1, names=["knot_num", "lon", "lat", "border_id"], sep='\s+', index_col=0)
    ds = ds.assign_coords(
        lon=("nodes_2d", df_lonlat_2d['lon'].values),
        lat=("nodes_2d", df_lonlat_2d['lat'].values),)
    ds.lon.attrs['description'] = 'longitude from nod2d.out'
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lat.attrs['description'] = 'latitude from nod2d.out'
    ds.lat.attrs['units'] = 'degrees_north'
            
    return ds
    

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
    
    # Check if ds still has a time dimension or if it was already pre selected
    if 'time' in ds.dims:
        # Get data for one time step
        data_raw = ds[var].isel(time=time_idx)
    else:
        # If no time dimension, use the data directly
        data_raw = ds[var]
    
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