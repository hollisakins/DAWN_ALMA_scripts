import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import Planck15
import astropy.units as u
import warnings
from photutils.aperture import CircularAperture, RectangularAperture, CircularAnnulus, aperture_photometry
import tqdm 
from copy import copy

####################################################################################################
######################################## Matplotlib Setup ##########################################
####################################################################################################

# I use the following parameters to make my plots consistent

mpl.rc('font',**{'family':'serif','monospace':['Palatino']})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'figure.dpi': 200,
                     'font.size': 9,
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'legend.frameon': False,
                     'figure.constrained_layout.use': True,
                     'xtick.top': True,
                     'ytick.right': True,
                     'image.origin': 'lower'})


####################################################################################################
######################################### Helpful functions ########################################
####################################################################################################

# The following functions makes it easy to plot an ellipse indicating the beam 
# size over a matplotlib.axes.Axes object

def plotBeam(ax, hdu, xy=(2,-2), kpc=False, fc='w', ec='k', alpha=1, linewidth=0.5):
    '''Custom function for plotting an ellipse indicating the ALMA beam size on a matplotlib.axes.Axes object. 
    Arguments: (* = required)
    * hdu (astropy.fits hdu object or custom image class object)
    - xy (tuple, xy coordinates of ellipse center)
    - kpc (boolean, whether to use BMAJ/BMIN in kpc or arcsec)

    Usage: 
        im = image(...)
        fig, ax = plt.subplots(1,1)
        ax.imshow(im.data, ...)
        ax.plotBeam(im, xy=(-2,2))
        fig.show()
    '''

    hdr = hdu.header
    
    Bmaj = dict(hdr)['BMAJ']*60*60
    Bmin = dict(hdr)['BMIN']*60*60
    Bpa = dict(hdr)['BPA']
    
    if kpc:
        Bmaj = dict(hdr)['BMAJ_KPC']
        Bmin = dict(hdr)['BMIN_KPC']
    
    e = mpl.patches.Ellipse(xy, height=Bmaj, width=Bmin, angle=-Bpa, zorder=10000, fc=fc, ec=ec, alpha=alpha, linewidth=linewidth)
    ax.add_patch(e)
    
setattr(mpl.axes.Axes, "plotBeam", plotBeam)


def get_filepath(line, obstype, weighting):
    '''Function to return the filepath for a given image. Images are specified by (line, obstype, weighting). 
    * line: targeted emission line (e.g. 'CII', 'OIII') 
    * obstype: type of observation (e.g. 'continuum', 'cube', or 'linemfs')
    * weighting: tclean weighting and uvtaper (e.g. 'natural_uv0.5') 
    '''
    global target_name

    if obstype=='continuum':
        f = f'Imaging/{line}/Continuum/{target_name}_{line}_continuum_{weighting}.fits'
    if obstype=='linemfs':
        f = f'Imaging/{line}/Line/{target_name}_{line}_linemfs_{weighting}.fits'
    if obstype=='cube':
        f = f'Imaging/{line}/Line/{target_name}_{line}_cube_{weighting}.fits'
    return f

####################################################################################################
######################################### ALMA Image Class #########################################
####################################################################################################


####################################### Parameter specification ###################################
# these are parameters that are common to the entire project

target_name = 'A1689-zD1' 
z_target = 7.132

# if using multiple different ms, specify the filepath for the image that all other images will align to (astrometrically)
centering_line = 'CII'
centering_image = get_filepath(centering_line,'continuum','natural_uv0.3')

# x0 and y0 specify center (in pixels, for the centering_image), which is used as the common center for all images
common_x0 = 130
common_y0 = 127


class image:
    '''Custom class for ALMA image objects.'''
    def __init__(self, line, obstype, weighting, correct_mu=True, chanmin=0, chanmax=70):
        self.obstype = obstype
        self.line = line
        self.weighting = weighting
        self.plane = 'image' # image.plane specifies whether we're in the image or source plane

        ### load in the data
        f = get_filepath(line, obstype, weighting)
        
        if obstype=='cube':
            h = fits.open(f)[0]
            self.data = h.data[0]
            self.size= np.shape(self.data)[1]
            self.nchans = np.shape(self.data)[0]
            
            h_res = fits.open(get_filepath(line,obstype,weighting+'_residual'))[0]
            self.residual = h_res.data[0]
            
            h_pb = fits.open(get_filepath(line,obstype,weighting+'_pb'))[0]
            self.pb = h_pb.data[0]
            
        else:
            h = fits.open(f)[0]
            self.data = h.data[0][0]
            self.size = np.shape(self.data)[0]
            
            if weighting.endswith('_psf'):
                t = weighting.replace('_psf','_residual')
            else:
                t = weighting+'_residual'
            h_res = fits.open(get_filepath(line,obstype,t))[0]
            self.residual = h_res.data[0][0]
            
            if weighting.endswith('_psf'):
                t = weighting.replace('_psf','_pb')
            else:
                t = weighting+'_pb'
            h_pb = fits.open(get_filepath(line,obstype,t))[0]
            self.pb = h_pb.data[0][0]
            
            
        ### save header
        self.header = h.header
        self.x0, self.y0 = common_x0, common_y0
        
        ### reproject the data into [CII] coordinate system
        if line != centering_line:
            self.reproject()
        
        ### construct error map
        if self.obstype=='cube':
            self.std = np.array([np.nanstd(self.residual[v]) for v in range(self.nchans)])
            self.std_map = np.array([np.zeros(shape=(self.size,self.size)) + self.std[v] for v in range(self.nchans)])
            self.error = self.std_map * np.sqrt(self.NPixPerBeam)
        else:
            self.std = np.nanstd(self.residual)
            self.std_map = np.zeros(shape=np.shape(self.data)) + self.std
            self.error = self.std_map * np.sqrt(self.NPixPerBeam)
        
        
        ### correct for magnification
        if correct_mu:
            dx, dy, mu = open_lens_model()
            self.data /= mu
            self.residual /= mu
            self.std_map /= mu
            self.error /= mu
            self.std /= np.mean(mu[self.dists < 2])
            
        ### undo primary beam correction for residual map
        self.std_map /= self.pb
        self.error /= self.pb
            
        ### for line mfs maps, convert units to moment 0 map
        if obstype=='linemfs':
            if line=='OIII':
                nu_min = 416.6 # GHz
                nu_max = 417.8 
                nu_rest = 417.2 
            elif line=='CII':
                nu_min = 233.4 # GHz
                nu_max = 234.0 
                nu_rest = 233.7
                
            from astropy.constants import c
            c = c.to(u.m/u.s).value
            v_max = c*(nu_rest - nu_min)/nu_rest
            v_min = c*(nu_rest - nu_max)/nu_rest
            self.delta_v = (v_max - v_min)/1000
            self.data *= self.delta_v
            self.residual *= self.delta_v
            self.std *= self.delta_v
            self.std_map *= self.delta_v
            self.error *= self.delta_v
        
        ### convert from Jy/beam to Jy/pix
        self.data /= self.NPixPerBeam
        self.residual /= self.NPixPerBeam
        self.std /= self.NPixPerBeam
        self.std_map /= self.NPixPerBeam
        self.error /= self.NPixPerBeam
    
    def reproject(self):
        '''Method to perform reprojection (astrometric alignment) on image object.'''
        from reproject import reproject_interp
        h_match = fits.open(centering_image)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs_match = WCS(h_match.header, naxis=2)
            wcs = WCS(self.header, naxis=2)
            if self.obstype=='cube':
                data = np.zeros(shape=np.shape(self.data))
                residual = np.zeros(shape=np.shape(self.data))
                for v in range(self.nchans):
                    d, footprint = reproject_interp((self.data[v],wcs), wcs_match, shape_out=(self.size,self.size))
                    data[v] = d
                    r, footprint = reproject_interp((self.residual[v],wcs), wcs_match, shape_out=(self.size,self.size))
                    residual[v] = r
            else:
                data, footprint = reproject_interp((self.data,wcs), wcs_match, shape_out=(self.size,self.size))
                residual, footprint = reproject_interp((self.residual,wcs), wcs_match, shape_out=(self.size,self.size))
                pb, footprint = reproject_interp((self.pb,wcs), wcs_match, shape_out=(self.size,self.size))

        self.header.update(wcs_match.to_header())
        self.data = data
        self.residual = residual
        
    
    @property
    def cell(self): 
        '''Cell (pixel) size in arcsec'''
        return np.abs(self.header['CDELT1']*60*60)
    
    @property
    def cell_kpc(self):
        '''Cell (pixel) size in kpc, after source plane correction'''
        if self.plane=='image':
            raise Exception('Image plane does not have well-defined pixel scale in kpc. Use image.cell instead.')
        elif self.plane=='source':
            from astropy.cosmology import Planck15
            scale = Planck15.kpc_proper_per_arcmin(z_target).to(u.kpc/u.arcsec).value
            return self.cell*scale

    @property
    def extent(self):
        '''Image extent (left,right,bottom,top), in arcsec from center, to pass to matplotlib imshow function'''
        left = self.x0*self.cell
        right = -(self.size-self.x0)*self.cell
        bottom = -self.y0*self.cell
        top = (self.size-self.y0)*self.cell
        return (left,right,bottom,top)
        
    @property
    def extent_kpc(self):
        '''Image extent (left,right,bottom,top), in kpc from center, to pass to matplotlib imshow function'''
        if self.plane=='image':
            raise Exception('Image plane does not have well-defined extent in kpc. Use image.extent instead.')
        elif self.plane=='source':
            left = -self.x0*self.cell_kpc
            right = (self.size-self.x0)*self.cell_kpc
            bottom = -self.y0*self.cell_kpc
            top = (self.size-self.y0)*self.cell_kpc
            return (left, right, bottom, top)
        
    @property
    def dists(self):
        '''2d array of distance from center, in arcsec, for each pixel'''
        x, y = np.arange(0, self.size, 1), np.arange(0, self.size, 1)
        x, y = np.meshgrid(x,y)
        return np.sqrt((x-self.x0)**2 + (y-self.y0)**2)*self.cell
        
    @property
    def BeamArea(self):
        '''Beam area in square arcseconds'''
        Bmaj = self.header['BMAJ']*60*60
        Bmin = self.header['BMIN']*60*60
        return np.pi/(4*np.log(2))*Bmaj*Bmin
    
    @property
    def NPixPerBeam(self):
        '''Number of pixels per beam'''
        return self.BeamArea/(self.cell**2)
        
    
    def RadialProfile(self, normalized=True, bins=np.arange(0.01, 4, 0.2), cutoff=True):
        '''Method to compute radial profiles from image object. 
        Returns (bincenters, surface brightness, surfacebrightness error). 
        Usage: 
            im = image(...)
            bc, sb, sb_err = im.RadialProfile(bins=np.arange(0.01, 10, 1))
            plt.errorbar(bc, sb, yerr=sb_err)
            
        Arguments: 
        * bins (array, default np.arange(0.01, 4, 0.2), radial profile bins in arcsec
        * normalized (bool, default True, whether to normalize to maximum)
        * cutoff (bool, default True, whether to cut off radial profile after first drops below zero)'''

        from photutils.aperture import CircularAnnulus, aperture_photometry

        x0, y0 = self.x0, self.y0
            
        ### construct aperture
        apertures = [CircularAnnulus([self.x0,self.y0], r_in=r_in/self.cell, r_out=r_out/self.cell) for r_in,r_out in zip(bins[:-1],bins[1:])]
        self.apertures = apertures
            
        ### setup data and error
        data = copy(self.data)
        error = copy(self.error)
        
        ### perform photometry 
        phot_table = aperture_photometry(data, apertures, error=self.error, mask=np.isnan(data))
        self.phot_table = phot_table
        area = np.array([a.area for a in apertures])
        sb = np.array([self.phot_table[f'aperture_sum_{i}'][0] for i in range(len(bins)-1)])/area
        sb_err = np.array([self.phot_table[f'aperture_sum_err_{i}'][0] for i in range(len(bins)-1)])/area
        
        if normalized:
            sb_err /= np.max(sb)
            sb /= np.max(sb)
            
        if cutoff:
            if any(sb < 0):
                i = np.min(np.arange(len(sb))[sb < 0])
                sb[i:] = 0
            
        bc = 0.5*(bins[1:]+bins[:-1])
        
        return bc, sb, sb_err
    
    
    # def BeamProfile(self, normalized=True, bins=np.arange(0.01, 16, 2.25)):
        #'''Function to compute profiles along beam shape'''
        # (removed, if need code for this contact Hollis)
        
    # def AxisProfile(self, axis, FWHM=None, boxwidth=3, theta=35, theta_source=-12, normalized=True, average=False, N=20, kpc=True):
        #'''Function to compute profiles along major/minor axis'''
        # (removed, if need code for this contact Hollis)
        
    
    def GrowthCurve(self, radii=np.arange(0.1, 6, 0.1)):
        '''Function to plot growth curve (cumulative radial proflie). 
        Arguments: 
        * radii (bins)
        '''

        from photutils.aperture import CircularAperture, aperture_photometry

        apertures = [CircularAperture([self.x0,self.y0], r/self.cell) for r in radii]

        error = self.total_error
        phot_table = aperture_photometry(self.data, apertures, error=error)
        self.phot_table = phot_table
        aperture_sum = np.array([self.phot_table[f'aperture_sum_{i}'][0] for i in range(len(radii))])
        aperture_sum_err = np.array([self.phot_table[f'aperture_sum_err_{i}'][0] for i in range(len(radii))])/np.sqrt(self.NPixPerBeam)
        
        aperture_sum_err /= aperture_sum[-1]
        aperture_sum /= aperture_sum[-1]
        
        return radii, aperture_sum, aperture_sum_err
    
    
    def Spectrum(self, aperture, restfreq=None, aperture_units='arcsec'):
        '''Function to compute the image spectrum (frequency [GHz], flux [mJy]) in an specified aperture.
           Arguments: 
           * aperture, specified as (x0,y0,R) where x0 = central right ascension in arcsec from center, 
             y0 = central declination in arcsec from center, R = radius in arcsec
             can also specify aperture as either a boolean mask (2d array) or a photutils.aperture object
           * restfreq, optional specification of central line frequency in GHz. 
             if not specified, x-axis units are in GHz. if specified, x-axis units are in km/s
           * aperture_units, either arcsec or pixels'''
        
        # if aperture is a tuple specifying (x0,y0,R)
        if type(aperture)==tuple:
            x0, y0, R = aperture
            if aperture_units=='arcsec':
                x0 = -x0/self.cell + self.x0
                y0 = y0/self.cell + self.y0
                R = R/self.cell
                aperture = CircularAperture([x0,y0], R)
                self.aperture_patch = mpl.patches.Circle((-(x0-self.x0)*self.cell, (y0-self.y0)*self.cell),
                                                         radius=R*self.cell, fc='none', ec='w', lw=0.8, zorder=2000)
            else:
                self.aperture_patch = mpl.patches.Circle((x0, y0), radius=R, fc='none', ec='w', lw=0.8, zorder=2000)
                    
            self.aperture = aperture

            flux, flux_err = np.zeros(shape=self.nchans),np.zeros(shape=self.nchans)
            for v in range(self.nchans):
                im = self.data[v,:,:]
                er = self.error[v,:,:]
                phot_table = aperture_photometry(im, aperture, error=er)
                flux[v] = np.array(phot_table['aperture_sum'])[0]*1000
                flux_err[v] = np.array(phot_table['aperture_sum_err'])[0]*1000
         
        # if aperture is a boolean array (a mask)
        elif type(aperture)==np.ndarray:
            flux = np.zeros(shape=self.nchans)
            for v in range(self.nchans):
                im = self.data[v,:,:]
                flux[v] = np.sum(im[aperture])*1000
        
        # if aperture is a photutils.aperture object
        else: 
            self.aperture = aperture

            flux = np.zeros(shape=self.nchans)
            for v in range(self.nchans):
                im = self.data[v,:,:]
                phot_table = aperture_photometry(im, aperture)
                flux[v] = np.array(phot_table['aperture_sum'])[0]*1000
        
        CDELT3 = self.header['CDELT3']
        CRVAL3 = self.header['CRVAL3']
        freq = (np.arange(0,self.nchans,1)-1)*CDELT3 + CRVAL3
        freq = freq / 1e9
        
        if restfreq==None:
            x = freq
        else:
            vel = 2.998e5 * (restfreq - freq)/restfreq
            x = vel
        
        return x, flux, flux_err
    
    
    def Reconstruct(self, cell=0.035, beam=False, f=np.nanmean):
        '''Function to performs source plane reconstruction and replace image with source-plane map.
        Arguments: 
        * cell: new pixel size in arcsec'''
        
        if not beam: 
            print('Performing source-plane reconstruction. Image-plane properties will be overwritten with source-plane properties.')
        
        dx, dy, mu = open_lens_model()
        x, y = np.arange(0, self.size, 1), np.arange(0, self.size, 1)
        x, y = np.meshgrid(x, y)
        x = (x - self.header['CRPIX1'])*self.header['CDELT1'] + self.header['CRVAL1']
        y = (y - self.header['CRPIX2'])*self.header['CDELT2'] + self.header['CRVAL2']
        
        x_source = x - dx
        y_source = y - dy
        
        crpix1 = self.size//2
        crpix2 = self.size//2
        crval1 = 197.87615818272903
        crval2 = -1.3371902006535314
        cdelt1 = cell/60/60
        cdelt2 = cell/60/60
        
        xbins = np.arange(0, self.size+1, 1)
        ybins = np.arange(0, self.size+1, 1)
        xbins = (xbins-crpix1)*cdelt1 + crval1
        ybins = (ybins-crpix2)*cdelt2 + crval2
        
        from scipy.stats import binned_statistic_2d
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self.obstype=='cube':
                x, y, z = x_source.flatten(), y_source.flatten(), self.mom0.flatten()
                values, xbins1, ybins1, binnumber = binned_statistic_2d(y, x, z, bins=(ybins, xbins), statistic=f)
                self.mom0 = values
            else:   
                x, y, z = x_source.flatten(), y_source.flatten(), copy(self.data).flatten()
                values, xbins1, ybins1, binnumber = binned_statistic_2d(y, x, z, bins=(ybins, xbins), statistic=f)
                self.data = values
                
                if not beam and self.obstype != 'HST':
                    z = copy(self.residual).flatten()
                    values, xbins1, ybins1, binnumber = binned_statistic_2d(y, x, z, bins=(ybins, xbins), statistic=f)
                    self.residual = values
                    
                    z = copy(self.std_map).flatten()
                    values, xbins1, ybins1, binnumber = binned_statistic_2d(y, x, z, bins=(ybins, xbins), statistic=f)
                    self.std_map = values
                    
                    z = copy(self.error).flatten()
                    values, xbins1, ybins1, binnumber = binned_statistic_2d(y, x, z, bins=(ybins, xbins), statistic=f)
                    self.error = values

        
        w = WCS(naxis=2)
        w.wcs.crpix = [-crpix1,crpix2]
        w.wcs.cdelt = np.array([-cdelt1, cdelt2])
        w.wcs.crval = [crval1, crval2]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        self.header.update(w.to_header())
        
        self.x0, self.y0 = self.size//2, self.size//2

        self.plane = 'source'
        if not beam:
            self.update_beam_size()
            
        
    def update_beam_size(self):
        # compute new beam size for reconstructed map
        Bmaj = self.header['Bmaj']*60*60/self.cell # in pixels
        Bmin = self.header['Bmin']*60*60/self.cell # in pixels
        theta = self.header['BPA']*np.pi/180 # in radians
        #print(f"Beam Size: {Bmaj*im.cell:.2f}'' x {Bmin*im.cell:.2f}'', BPA = {theta/np.pi*180:.2f} degrees")

        sigma_maj = Bmaj / (2*np.sqrt(2*np.log(2)))
        sigma_min = Bmin / (2*np.sqrt(2*np.log(2)))

        # produce gaussian model for the beam
        from astropy.convolution import Gaussian2DKernel
        beam_model = Gaussian2DKernel(sigma_maj,sigma_min,theta*np.pi/180-np.pi/2, x_size=self.size, y_size=self.size)
        
        try:
            im = image(self.line, self.obstype, self.weighting)
        except AttributeError: # for HST images
            im = image(self.psfmatch[0], self.psfmatch[1], self.psfmatch[2])
            
        
        im.data = beam_model.array # set psf array to the gaussian model for the psf (simpler!)
        im.Reconstruct(beam=True) # use beam=True to tell Reconstruct to not run update_beam_size again
        im.data[np.isnan(im.data)] = 0 # remove nan values from resulting reconstructed image
        
        # fit a Gaussian to the source-plane beam
        from astropy.modeling import models, fitting

        p_init = models.Gaussian2D(amplitude=1, x_mean=im.x0, y_mean=im.y0, x_stddev=0.5/im.cell, y_stddev=0.1/im.cell, theta=90/180*np.pi)
        fit_p = fitting.LevMarLSQFitter()
        x, y = np.arange(0, im.size, 1), np.arange(0, im.size, 1)
        x, y = np.meshgrid(x, y)
        p = fit_p(p_init, x, y, im.data, maxiter=1000)
        
        self.header['Bmaj'] = p.y_stddev.value * im.cell * 2.355 / 60 /60
        self.header['Bmin'] = p.x_stddev.value * im.cell * 2.355 / 60 /60
        self.header['BPA'] = p.theta.value*180/np.pi - 180
        
        self.header['Bmaj_kpc'] = p.y_stddev.value * im.cell_kpc * 2.355 
        self.header['Bmin_kpc'] = p.x_stddev.value * im.cell_kpc * 2.355 
            

def reproject_HST(filt):
    # Step 1: Load reduced HST image
    filt = filt.lower()
    hdu = fits.open(f'Data/HST/a1689-j1311m0120-{filt}_drz_sci.fits')[0]
    wcs = WCS(hdu.header)

    # Step 2: Reproject
    hdu_convolve = fits.open('Imaging/CII/Continuum/A1689-zD1_CII_continuum_natural_uv0.fits')[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs2 = WCS(hdu_convolve.header, naxis=2)
        from reproject import reproject_exact
        hdu.data, footprint = reproject_exact((hdu.data, wcs), wcs2, shape_out=np.shape(hdu_convolve.data[0][0]))
        hdu.header.update(wcs2.to_header())

    filename = f'Data/HST/a1689-j1311m0120-{filt}_drz_sci_reproject.fits'
    print('Writing to',filename)
    hdu.writeto(filename, overwrite=True)

        
class HSTimage(image):
    def __init__(self, filt, mask=True, convolve=True, correct_mu=True, psfmatch=('OIII','continuum','natural_uv0'), nan_treatment='interpolate'):
        self.filt = filt
        self.plane = 'image'
        self.obstype = 'HST'
        self.psfmatch = psfmatch
        z = 7.132
        dl = Planck15.luminosity_distance(z).to(u.cm)
        from astropy.constants import c
        c = c.to(u.m/u.s)
        self.dl = dl
        
        # Step 1: Load reduced HST image
        filt = filt.lower()
        hdu = fits.open(f'Data/HST/a1689-j1311m0120-{filt}_drz_sci_reproject.fits')[0]
        PHOTFNU = hdu.header['PHOTFNU'] # inverse sensitivity, Jy s/electron
        array = hdu.data*PHOTFNU*1e-23 # now in units of erg/s/cm^2/Hz

        if filt=='f125w':
            lam = 12364.65 * u.Angstrom
        elif filt=='f160w':
            lam = 15279.08 * u.Angstrom
        else: raise Exception('incorrect filter specification');
        
        nu = c/lam
        nu = nu.to(u.Hz)
        nu_rest = nu.value * (z+1)
        self.nu_rest = nu_rest
        
        array = array / (1+z) * 4*np.pi*(dl.to(u.cm).value)**2 * nu_rest # now in units of erg/s
        
        if correct_mu:
            dx, dy, mu = open_lens_model()
            array = array / mu
            
        self.header = hdu.header
        self.header['BUNIT'] = 'erg/s'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs = WCS(self.header)
            
        self.size = np.shape(array)[0]
        self.x0, self.y0 = 130,127

        # Step 3: Background subtract
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MedianBackground()
        bkg = Background2D(array, (20, 20), filter_size=(7, 7),
                           sigma_clip=sigma_clip, 
                           bkg_estimator=bkg_estimator)
        self.bkg = bkg
        array = array - bkg.background
        if mask:
            m = (self.dists > 1.1)#&(self.dists < 5/self.cell)
            for i in range(10):
                median, std = np.nanmedian(array[m]), np.nanstd(array[m])
                array[m & ((array > median+2.5*std)|(array < median-2.5*std))] = np.nan
                
            np.random.seed(312)
            rand = np.random.normal(loc=median, scale=std, size=(self.size,self.size))
            array[np.isnan(array)] = rand[np.isnan(array)]
            
            x0, y0 = 158, 115
            x, y = np.arange(0, self.size, 1), np.arange(0, self.size, 1)
            x, y = np.meshgrid(x, y)
            dists = np.sqrt((x-x0)**2 + (y-y0)**2)
            array[dists < 14] = rand[dists < 14]

        # Step 5: Convolve both arrays with ALMA beam
        self.psfmatch = psfmatch
        if convolve:
            from astropy.convolution import convolve
            array = convolve(array, self.matching_kernel, nan_treatment='fill', fill_value=0)
        
        self.data = array
        
        self.std = np.std(self.data[(self.dists > 1.1)&(self.dists < 3)])
        self.error = np.zeros(shape=np.shape(self.data)) + self.std
        
        self.sfr = array * 4.42e-44
        self.sfr_err = self.error * 4.42e-44
        
    

    
    @property
    def matching_kernel(self):
        from astropy.convolution import Gaussian2DKernel
        
        # Construct Kernel for HST PSF (PSF values from Windhorst et al. 2011)
        if self.filt=='F160W':
            FWHM = 0.15 # arcsec
        elif self.filt=='F125W':
            FWHM = 0.136 # arcsec
        else:
            raise Exception(f'No PSF FWHM has been defined for filter {self.filt}')
        
        FWHM = FWHM/self.cell
        sigma = FWHM / (2*np.sqrt(2*np.log(2)))
        theta = 0
        k_hst = Gaussian2DKernel(sigma,sigma,theta, x_size=201, y_size=201)
        
        line, obstype, weighting = self.psfmatch
        weighting = weighting+'_psf'
        hdu = image(line, obstype, weighting)
        
        Bmaj = hdu.header['BMAJ']*60*60/hdu.cell
        Bmin = hdu.header['BMIN']*60*60/hdu.cell
        sigma_maj = Bmaj / (2*np.sqrt(2*np.log(2)))
        sigma_min = Bmin / (2*np.sqrt(2*np.log(2)))
        theta = hdu.header['BPA']
        
        self.header['BMAJ'] = Bmaj/60/60*hdu.cell
        self.header['BMIN'] = Bmin/60/60*hdu.cell
        self.header['BPA'] = theta
        

        k_alma = Gaussian2DKernel(sigma_maj,sigma_min,theta*np.pi/180-np.pi/2, x_size=201, y_size=201)
        
        from photutils.psf import TopHatWindow, create_matching_kernel
        window = TopHatWindow(0.35)
        kernel = create_matching_kernel(k_hst, k_alma, window=window)        
        
        return kernel
    
    @property
    def BeamArea(self):
        return 1
    
    @property
    def NPixPerBeam(self):
        return 1
    

def open_lens_model(scale_factor = 0.913):
    '''Custom function for opening the lens model. 
    Requires that you place the lens model (dx,dy,gamma,kappa).fits in a directory called LensModel/
    Returns (dx [degrees], dy [degrees], mu). '''
    dx = fits.open('LensModel/dx.fits')[0]
    dy = fits.open('LensModel/dy.fits')[0]
    gamma = fits.open('LensModel/gamma.fits')[0]
    kappa = fits.open('LensModel/kappa.fits')[0]
    dx.data *= scale_factor
    dy.data *= scale_factor
    gamma.data *= scale_factor
    kappa.data *= scale_factor
    
    from reproject import reproject_interp
    hdu = fits.open(centering_image)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = WCS(hdu.header, naxis=2)
        s = np.shape(hdu.data[0,0,:,:])

        dx_r, footprint = reproject_interp(dx, wcs, shape_out=s)
        dy_r, footprint = reproject_interp(dy, wcs, shape_out=s)
        kappa_r, footprint = reproject_interp(kappa, wcs, shape_out=s)
        gamma_r, footprint = reproject_interp(gamma, wcs, shape_out=s)
    
    mu_inverse = (1-kappa_r)**2 - gamma_r**2
    mu = 1/mu_inverse
    
    dx_r = -dx_r/60/60
    dy_r = dy_r/60/60
    return dx_r, dy_r, mu