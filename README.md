
The basic structure of my code is: 
- a class called `image` that represents ALMA images. 
    - functions that act on this `image` class, such as `RadialProfile` and `Spectrum`
- a sub-class called `HSTimage` that represents HST images

These classes are specified in `base.py`, which must be imported before analysis. 
To perform analysis, one can load in an image and run functions as so: 

```
from base import *
im = image('CII','continuum','natural_uv0')
bc, sb, sb_err = im.RadialProfile(...)

im = image('CII','cube','natural_uv0')
nu, flux, flux_err = im.Spectrum(...)
```

The `image` class expects that you have exported your CASA images as `.fits` files and saved them under the following directory structure: 
```
arbitrary_notebook.ipynb
arbitrary_script.py 
base.py 
Imaging/
    CII/
        Line/
            {target_name}_CII_linemfs_natural_uv0.fits
            {target_name}_CII_linemfs_natural_uv0_residual.fits
            {target_name}_CII_linemfs_natural_uv0_psf.fits
            {target_name}_CII_linemfs_natural_uv0_pb.fits
        Continuum/
            ...
    OIII/
        ...
```
The basic filenaming convention is `{target_name}_{line}_{obstype}_{weighting}.fits`.
Note: not only do you need to export the CASA `.image.pbcor` file as the base `.fits` file, but you also need to export the `.residual`, `.psf`, and `.pb` files as above.

There are a number of parameters that you will want to specify within `base.py`. 
You'll need to specify the target name and redshift, as well as the common center `x` and `y` coordiantes (in pixels) for the images. 
If working with multiwavelength data (i.e. multiple ALMA measurement sets), you'll want to ensure that each `ms` is astrometrically aligned. 
To do that, you can specify a path to one of your `.fits` files in `base.py` under `centering_image= ...`. 
This will ensure that all other images are aligned to this one using the python `reproject` package. 