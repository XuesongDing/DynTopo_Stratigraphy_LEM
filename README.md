# GRL input


Supporting information that contains all the input and forcing conditions files required to reproduce the experiments from the manusript: **Drainage and sedimentary responses to dynamic topography** (https://doi.org/10.1029/2019GL084400).


<div align="center">
    <img width=700 src="https://github.com/XuesongDing/DynTopo_Stratigraphy_LEM/blob/master/images/Results/fig2.jpg" alt="Predicted stratal architecture from pyBadlands" title="Stratigraphic responses to dynamic topography wave"</img>
</div>

This works applies the landscape evolution model [**Badlands**](https://github.com/badlands-model/pyBadlands) as a numerical tool to investigate the contribution of dynamic topography to continental-scale drainage evolution and the formation of stratigraphic architecture at passive continental margins. 


You will need to download and install <a href='https://github.com/badlands-model/pyBadlands/releases' target="_blank">Badlands v2.0.0<a/> to run these experiments.

[![DOI](https://zenodo.org/badge/51286954.svg)](https://zenodo.org/badge/latestdoi/51286954)

An easy installation is provided through our Docker image _(pyBadlands-demo-serial image)_.

## Content

+ **data**: This folder contains the node file for initial surface and the sea-level file. 
+ **images**: This folder contains images used in post-processing scripts and of the results.
+ **scripts**: Post-processing scripts to create the dynamic topography wave and to visualize predicted stratigraphic architectures. 
+ **xmL files**: Main entry point for defining the initial and forcing conditions used by **Badlands** to run any model (this is required).
+ **Run_pyBadlands.ipynb**: Ipython notebook to run Badlands models.
