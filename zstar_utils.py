'''
Created by Júlia Kaiser - Mar/2026
CMCC Foundation - Euro-Mediterranean Center on Climate Change
GOCO - Global Coastal Ocean Division
'''

def Z_to_Zstar_bathyfix(Zlayers, bathymetry, eta, ilytyp, hlvmin):
    """
    This function converts Z layers to Z* coordinates with fixed bathymetry.

    Inputs are:
    Zlayers                     - Vertical layer interfaces (1D array)
    bathymetry                  - Ponctual water depth (float)
    eta                         - Free surface elevation (time-dependent)
    ilytyp                      - Treatmenet of the bottom layer from SHYFEM namelist (int)
    hlvmin                      - Minimum layer thickness for the bottom layer when ilytyp is 2 or 3 (int)
    
    Outputs are:
    h_zstar                     - Layer thickness in Z* coordinates (time-dependent)
    Zstar_interfaces            - Interface depths (bottom) in Z* coordinates (time-dependent)
    Zinterfaces_ZstarSystem     - Z layers position in the Z* coordinate system (time-dependent) -- consistent with Giorgia's matlab code (and SHYFEM code?)
    """

    import numpy as np

    # --- Reference layer thicknesses ---
    z_interfaces = Zlayers
    hlv = np.diff(np.concatenate(([0.0], z_interfaces)))  # (nlv,)

    # --- Active layers ---
    act_layer = np.where(z_interfaces <= float(bathymetry))[0] # We are considering fixed bathymetry for each time step

    if ilytyp == 3:
        z_nominal_interfaces = z_interfaces[act_layer]
        nominal_hlv = hlv[act_layer]

        diff_bathy = bathymetry - z_nominal_interfaces[-1]
        thick_threshold = hlv[act_layer[-1] + 1] * hlvmin

        if diff_bathy < thick_threshold:
            z_nominal_interfaces[-1] = bathymetry
            nominal_hlv[-1] = nominal_hlv[-1] + diff_bathy
        else: #if diff_bathy >= thick_threshold:
            z_nominal_interfaces = np.append(z_nominal_interfaces, bathymetry)
            nominal_hlv = np.append(nominal_hlv, z_nominal_interfaces[-1] - z_nominal_interfaces[-2])

    # Z* scaling factor (alpha = (Htot + eta) / Htot)
    H = float(bathymetry)
    alpha = (H + eta) / H  # (time,)

    # Z* layer thickness
    h_zstar = alpha[:, None] * nominal_hlv[None, :]  # (time, level)

    # Z* level bottom interfaces (cumulative)
    Zstar_interfaces = np.cumsum(h_zstar, axis=-1) # Standard Z* interfaces (no surface correction)

    # ---
    # Getting Z layers position in Z* coordinate system (Zinterfaces_ZstarSystem)
    # ---
    
    #Zinterfaces_ZstarSystem = Zstar_interfaces - eta[:, None]
    
    # 1. Start from SHYFEM surface: z[0] = -eta
    Zinterfaces_ZstarSystem = np.zeros_like(Zstar_interfaces)

    # Surface
    Zinterfaces_ZstarSystem[:, 0] = -eta

    # First layer correction (SHYFEM-specific)
    #Zinterfaces_ZstarSystem[:, 1] = (Zstar_interfaces[:, 0] + h_zstar[:, 0] - eta)   # first thickness
    Zinterfaces_ZstarSystem[:, 1] = ( Zinterfaces_ZstarSystem[:, 0] + h_zstar[:, 0] - eta )
    
    # 3. Remaining layers (TRUE cumulative from layer 1 onward)
    if h_zstar.shape[1] > 2:
        #Zinterfaces_ZstarSystem[:, 2:] = ( Zinterfaces_ZstarSystem[:, 1][:, None] + np.cumsum(h_zstar[:, 1:-1], axis=1) )
        Zinterfaces_ZstarSystem[:, 2:] = ( Zinterfaces_ZstarSystem[:, 1][:, None] + np.cumsum(h_zstar[:, 1:], axis=1)[:, :-1] )
    
    # Hard-coding the solution for the last layer (not ideal!)
    Zinterfaces_ZstarSystem[:, -1] = bathymetry - eta
    
    return h_zstar, Zstar_interfaces, Zinterfaces_ZstarSystem



def Z_to_Zstar_bathyvar(Zlayers, bathymetry, eta, ilytyp, hlvmin):
    """
    Z* conversion for variable bathymetry (field: time × space).

    Parameters
    ----------
    Zlayers : (nlevels,)
    bathymetry : (npoints,)
    eta : (ntime, npoints)
    ilytyp : int
    hlvmin : float

    Returns
    -------
    h_zstar : (ntime, npoints, nlevels)
    Zstar_interfaces : (ntime, npoints, nlevels)
    Zinterfaces_ZstarSystem : (ntime, npoints, nlevels)
    """

    import numpy as np
    
    # --- Reference layer thicknesses ---
    z_interfaces = Zlayers
    hlv = np.diff(np.concatenate(([0.0], z_interfaces)))
    
    # --- Adjusting variables dimensions ---
    ntime, npoints = eta.shape
    max_levels = len(z_interfaces) + 1 # Maximum possible levels (including potential extra layer for ilytyp=3)
    
    bathymetry = np.asarray(bathymetry, dtype=float)
    eta = np.asarray(eta, dtype=float)

    # Normalize dimensions: eta (ntime, npoints), bathymetry (ntime, npoints)
    if eta.ndim == 1:
        eta = eta[:, None]
    if bathymetry.ndim == 1:
        bathymetry = bathymetry[None, :]

    if bathymetry.shape[0] == 1 and eta.shape[0] > 1:
        bathymetry = np.broadcast_to(bathymetry, (eta.shape[0], bathymetry.shape[1]))

    if bathymetry.shape != eta.shape:
        raise ValueError("eta and bathymetry must have compatible shapes")

    ntime, nloc = eta.shape

    # --- Allocate outputs ---
    h_zstar = np.full((ntime, npoints, max_levels), np.nan, dtype=float)
    Zstar_interfaces = np.full((ntime, npoints, max_levels), np.nan, dtype=float)
    Zinterfaces_ZstarSystem = np.full((ntime, npoints, max_levels), np.nan, dtype=float)


    # =========================================================
    # Loop over space (each point has different vertical grid)
    # =========================================================
    for ip in range(npoints):
        H = float(bathymetry[ip])
        if H <= 0: #inland point with no active layers
            continue

        # --- Active layers ---
        act_layer = np.where(z_interfaces <= H)[0]
        if len(act_layer) == 0: #point with no active layer
            continue

        # --- Nominal layers ---
        z_nominal_interfaces = z_interfaces[act_layer].copy()
        nominal_hlv = hlv[act_layer].copy()

        if ilytyp == 3:
            diff_bathy = H - z_nominal_interfaces[-1]

            if act_layer[-1] + 1 < len(hlv):
                thick_threshold = hlv[act_layer[-1] + 1] * hlvmin
            else:
                thick_threshold = hlv[-1] * hlvmin

            if diff_bathy < thick_threshold:
                z_nominal_interfaces[-1] = H
                nominal_hlv[-1] = nominal_hlv[-1] + diff_bathy
            else:
                z_nominal_interfaces = np.append(z_nominal_interfaces, H)
                nominal_hlv = np.append(nominal_hlv, z_nominal_interfaces[-1] - z_nominal_interfaces[-2])

        nlev_loc = len(nominal_hlv)

        # --- Time-dependent scaling ---
        alpha = (H + eta[:, ip]) / H  # (ntime,)

        # --- Thickness ---
        h_zstar = alpha[:, None] * nominal_hlv[None, :]  # (ntime, nlev_loc)

        # --- Z* interfaces ---
        Zstar_interfaces = np.cumsum(h_zstar, axis=1)

        # --- Zinterfaces in Z* system ---
        Zinterfaces_ZstarSystem = np.zeros_like(Zstar_interfaces)

        # Surface (for SHYFEM consistency)
        Zinterfaces_ZstarSystem[:, 0] = -eta[:, ip]

        # First layer
        if nlev_loc > 1:
            Zinterfaces_ZstarSystem[:, 1] = Zinterfaces_ZstarSystem[:, 0] + h_zstar[:, 0] - eta[:, ip]

        # Remaining layers
        if nlev_loc > 2:
            Zinterfaces_ZstarSystem[:, 2:] = (Zinterfaces_ZstarSystem[:, 1][:, None] + np.cumsum(h_zstar[:, 1:], axis=1)[:, :-1])

        # Bottom condition (SHYFEM-consistent; hard-coded is not ideal!!)
        Zinterfaces_ZstarSystem[:, -1] = H - eta[:, ip]

    return h_zstar, Zstar_interfaces, Zinterfaces_ZstarSystem
