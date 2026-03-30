This repository is inteded to store a set of scripts for post-processing SHYFEM unstructured output files.

1. zstar_utils.py
In the current version of SHYFEM, Z* parameters (such as layer thicknesses and interfaces) are not directly available in the output files.
Therefore, these information must be computed as a post-processing step.
This step is important for all vertical coordinate systems, including Z-level configurations, since the levels variable in SHYFEM outputs refers to the prescribed layers defined in the namelist.
However, it is especially critical for the Z* system, where vertical layers vary both in space and time.
So, this script compute Z* layers for a single point (Z_to_Zstar_bathyfix) and for the full computational domain (Z_to_Zstar_bathyvar).

The inputs from the user are: Z layers (namelist configuration), bathymetry, SSH (eta), ilytyp and hlvmin (both from namelist configuration setting).
The outputs are: Z* time-dependent layer thicknesses (h_zstar), Z* vertical interfaces (Zstar_interfaces), and corresponding Z interfaces within the Z* framework (Zinterfaces_ZstarSystem).

To run the script, the following Python packages are needed: numpy.
