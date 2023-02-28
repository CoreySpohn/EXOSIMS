from astropy.io import fits
import os
from EXOSIMS.util.radialfun import (
    radial_average,
    genwindow,
    resample_image,
    circ_aperture,
    com,
    fitgaussian,
)
from scipy.ndimage import shift
import numpy as np
import warnings
import json
import astropy.units as u


def process_opticalsys_package(
    basepath,
    stellar_intensity_file="stellar_intens.fits",
    stellar_intensity_diameter_list_file="stellar_intens_diam_list.fits",
    offax_psf_file="offax_psf.fits",
    offax_psf_offset_list_file="offax_psf_offset_list.fits",
    sky_trans_file="sky_trans.fits",
    resamp=4,
    outpath=None,
    outname="",
    name=None,
    phot_aperture_radius=np.sqrt(2) / 2,
    fit_gaussian=False,
    use_phot_aperture_as_min=False,
    overwrite=True,
):
    """Process optical system package defined by Stark & Krist to EXOSIMS
    standard inputs.

    Args:
        basepath (str):
            Full path to directory with all input files.
        stellar_intensity_file (str, optional):
            Filename of stellar intensity PSFs. Defaults to "stellar_intens.fits".
            If None, no output generated.
        stellar_intensity_diameter_list_file (str, optional):
            Filename of stellar diameters corresponding to stellar_intensity_file.
            Defaults to "stellar_intens_diam_list.fits". If None, no output generated.
        offax_psf_file (str):
            Filename of off-axis PSFs. Defaults to "offax_psf.fits".
            If None, no output generated.
        offax_psf_offset_list_file (str, optional):
            Filename of off-axis PSF astrophysical offsets corresponding to
            offax_psf_file. Defaults to "offax_psf_offset_list.fits".
            If None, no output generated.
        sky_trans_file (str, optional):
            Filename of sky transmission map.  Defualts to "sky_trans.fits"
            If None, no output generated.
        resamp (float):
            Resampling factor for PSFs.  Defaults to 4.
        outpath (str, optional):
            Full path to directory to write results. If None, use basepath.
            Defaults None
        outname (str):
            Prefix for all output files.  If "" (default) then no prefix is added,
            otherwise all files will be named outname_(descriptor).fits.
        name (str, optional):
            System name (for dictionary output). If None and outname is not "" then
            use outname.  If None and outname is "" then write None.  Defaults None.
        phot_aperture_radius (float):
            Photometric aperture radius in native unit of input files (nominally
            :math:`\\lambda/D`). Defaults to :math:`\\sqrt{2}/2`.  If fit_gaussian is
            True and use_phot_aperture_as_min is True then this value replaces any fit
            area that is smaller than the area of this value.
        fit_gaussian (bool):
            Fit 2D Gaussians to off-axis PSFs to compute throughput and core area.
            Default False, in which case the core_area is always the value computed from
            phot_aperture_radius.
        use_phot_aperture_as_min (bool):
            Only used if fit_gaussian is True.  If True, any coputed core area values
            that are smaller than the area of the phot_aperture_radius are replaced with
            that value. Defaults False
        overwrite (bool):
            Overwrite output files if they exist. Defaults True. If False, throws error.

    Returns:
        dict:
            :ref:`StarlightSuppressionSystem` dictionary describing the generated
            system

    .. note::

        The default expectation is that all 5 input files will be provided and
        processed. However, any of these can be set to None, in which case processing
        will be skipped for that filetype.  Paired files (i.e., ``intens`` and
        ``intens_diam`` will be skipped if either is None).

    """
    if outpath is None:
        outpath = basepath
    if name is None and outname != "":
        name = outname
    if outname != "":
        outname += "_"

    # package inputs
    ftypes = ["intens", "intens_diam", "offax_psf", "offax_psf_offset", "sky_trans"]
    fnames = [
        stellar_intensity_file,
        stellar_intensity_diameter_list_file,
        offax_psf_file,
        offax_psf_offset_list_file,
        sky_trans_file,
    ]

    headers = {}
    data = {}
    for ftype, fname in zip(ftypes, fnames):
        if fname is not None:
            with fits.open(os.path.join(basepath, fname)) as hdulist:
                data[ftype] = hdulist[0].data
                headers[ftype] = hdulist[0].header
        else:
            data[ftype] = None
            headers[ftype] = None

    # keep track of header values
    header_vals, IWA, OWA = [None] * 3

    # Sky (Occulter) Transmission Map: tau_occ (T_sky):
    if data["sky_trans"] is not None:
        occ_trans_vals, _, bc = radial_average(
            data["sky_trans"],
            center=[headers["sky_trans"]["XCENTER"], headers["sky_trans"]["YCENTER"]],
        )
        occ_trans_wa = bc * headers["sky_trans"]["pixscale"]

        # write to disk
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(
                    np.vstack((occ_trans_wa, occ_trans_vals)).transpose(),
                    header=headers["sky_trans"],
                )
            ]
        )
        occ_trans_fname = os.path.join(outpath, f"{outname}occ_trans.fits")
        hdul.writeto(occ_trans_fname, overwrite=overwrite)
        header_vals = check_header_vals(header_vals, headers["sky_trans"])
        IWA, OWA = update_WA_vals(IWA, OWA, occ_trans_wa)
    else:
        occ_trans_fname = None

    # Off-Axis PSF maps: core_thruput, tau_core (Upsilon)
    if (data["offax_psf_offset"] is not None) and (data["offax_psf"] is not None):
        # pixel scale must be the same in both files
        assert (
            headers["offax_psf_offset"]["PIXSCALE"] == headers["offax_psf"]["PIXSCALE"]
        ), "PIXSCALE in offax_psf and offax_psf_offset files is different."

        # data must be two-dimensional:
        assert (
            2 in data["offax_psf_offset"].shape
        ), "offax psf offset data must be 2-dimensional"
        if data["offax_psf_offset"].shape[0] != 2:
            data["offax_psf_offset"] = data["offax_psf_offset"].transpose()

        # grab astrophysical source centers in pixel units
        acents = data["offax_psf_offset"] / headers["offax_psf_offset"]["PIXSCALE"]

        # grab image dimensions and generate resampled window
        dims = data["offax_psf"][0].shape
        window = genwindow([(d - 1) * resamp + 1 for d in dims])
        # get photometric aperture radius in pixel units
        rho = phot_aperture_radius / headers["offax_psf_offset"]["PIXSCALE"]
        rho_resampled = rho * resamp

        # allocate storage arrays
        core_thruput_vals = np.zeros(len(data["offax_psf"]))
        cents = np.zeros((len(data["offax_psf"]), 2))
        if fit_gaussian:
            core_areas = np.zeros(core_thruput_vals.shape)

        for j in range(len(data["offax_psf"])):
            # resample image and apply window around astrophysical offset
            im = resample_image(data["offax_psf"][j], resamp=resamp)
            windim = im * shift((window), acents[::-1, j] * resamp)
            center = com(windim)
            cents[j] = center

            if fit_gaussian:
                g = fitgaussian(windim)
                # compute half-width at half max (NB: this is in rescaled pixel units)
                hwhm = np.mean(np.abs(g[-2:] * np.sqrt(2 * np.log(2))))
                if use_phot_aperture_as_min and (hwhm < rho_resampled):
                    hwhm = rho_resampled
                core_thruput_vals[j] = circ_aperture(
                    im, hwhm, center, return_sum=True
                ) / (resamp**2)
                core_areas[j] = (
                    np.pi
                    * (hwhm / resamp * headers["offax_psf_offset"]["PIXSCALE"]) ** 2
                )
            else:
                core_thruput_vals[j] = circ_aperture(
                    im, rho_resampled, center, return_sum=True
                ) / (resamp**2)

        # now figure out the angles
        # check if either dimension contains just one value
        if np.all(data["offax_psf_offset"][0] == data["offax_psf_offset"][0][0]):
            core_thruput_wa = data["offax_psf_offset"][1]
        elif np.all(data["offax_psf_offset"][1] == data["offax_psf_offset"][1][0]):
            core_thruput_wa = data["offax_psf_offset"][0]
        else:
            core_thruput_wa = np.sqrt(
                data["offax_psf_offset"][0] ** 2 + data["offax_psf_offset"][1] ** 2
            )
        assert not (
            np.all(core_thruput_wa == core_thruput_wa[0])
        ), "Could not determine offax psf offsets"

        # write to disk
        hdr = headers["offax_psf_offset"].copy()
        if fit_gaussian:
            hdr["PHOTAPER"] = "Gaussian"
            if use_phot_aperture_as_min:
                hdr["MINAPER"] = phot_aperture_radius
            else:
                hdr["MINAPER"] = 0
        else:
            hdr["PHOTAPER"] = phot_aperture_radius

        # core throughput:
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(
                    np.vstack((core_thruput_wa, core_thruput_vals)).transpose(),
                    header=hdr,
                )
            ]
        )
        core_thruput_fname = os.path.join(outpath, f"{outname}core_thruput.fits")
        hdul.writeto(core_thruput_fname, overwrite=overwrite)

        # core area:
        if fit_gaussian:
            hdul = fits.HDUList(
                [
                    fits.PrimaryHDU(
                        np.vstack((core_thruput_wa, core_areas)).transpose(),
                        header=hdr,
                    )
                ]
            )
            core_area_fname = os.path.join(outpath, f"{outname}core_area.fits")
            hdul.writeto(core_area_fname, overwrite=overwrite)
        else:
            # if a fixed aperture was used, just write the scalar value
            core_area_fname = phot_aperture_radius**2 * np.pi

        header_vals = check_header_vals(header_vals, headers["offax_psf"])
        header_vals = check_header_vals(header_vals, headers["offax_psf_offset"])
        IWA, OWA = update_WA_vals(IWA, OWA, core_thruput_wa)
    else:
        core_thruput_fname = None
        core_area_fname = None

    # Stellar intensity maps: core_mean_intensity (I_xy)
    if (data["intens"] is not None) and (data["intens_diam"] is not None):
        # pixel scale must be the same in both files
        assert (
            headers["intens"]["PIXSCALE"] == headers["intens_diam"]["PIXSCALE"]
        ), "PIXSCALE in intens and intens_diam files is different."

        # stellar diameter list must be the same length as data
        assert len(data["intens_diam"]) == len(
            data["intens"]
        ), "intens and intens_diam must have the same length"

        # get first radial profile
        radintens, _, bc = radial_average(
            data["intens"][0],
            center=[headers["intens"]["XCENTER"], headers["intens"]["YCENTER"]],
        )

        # allocate output
        mean_intens = np.zeros((len(data["intens"]), len(radintens)))
        mean_intens[0] = radintens

        for j in range(1, len(data["intens"])):
            mean_intens[j], _, _ = radial_average(
                data["intens"][j],
                center=[headers["intens"]["XCENTER"], headers["intens"]["YCENTER"]],
            )

        mean_intens_wa = bc * headers["intens"]["pixscale"]
        out = np.vstack((mean_intens_wa, mean_intens))
        hdr = headers["intens_diam"]
        for j, val in enumerate(data["intens_diam"]):
            hdr[f"DIAM{j :03d}"] = val

        # write out:
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(
                    out.transpose(),
                    header=hdr,
                )
            ]
        )

        core_mean_intensity_fname = os.path.join(
            outpath, f"{outname}core_mean_intensity.fits"
        )
        hdul.writeto(core_mean_intensity_fname, overwrite=overwrite)

        header_vals = check_header_vals(header_vals, headers["intens"])
        header_vals = check_header_vals(header_vals, headers["intens_diam"])
        IWA, OWA = update_WA_vals(IWA, OWA, mean_intens_wa)
    else:
        core_mean_intensity_fname = None

    angunit = ((header_vals["lam"] * u.nm) / (header_vals["pupilDiam"] * u.m)).to(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    outdict = {
        "pupilDiam": header_vals["pupilDiam"],
        "obscurFac": header_vals["obscurFac"],
        "shapeFac": np.pi / 4,
        "starlightSuppressionSystems": [
            {
                "name": name,
                "lam": header_vals["lam"],
                "deltaLam": header_vals["deltaLam"],
                "occ_trans": occ_trans_fname,
                "core_thruput": core_thruput_fname,
                "core_mean_intensity": core_mean_intensity_fname,
                "core_area": core_area_fname,
                "IWA": (IWA * angunit).to(u.arcsec).value,
                "OWA": (OWA * angunit).to(u.arcsec).value,
            }
        ],
    }

    script_fname = os.path.join(outpath, f"{outname}_specs.json")
    with open(script_fname, "w") as f:
        json.dump(outdict, f)

    return outdict


def check_header_vals(header_vals, hdr):
    """Utility method for checking values in headers

    Args:
        header_vals (dict, optional):
            Current value set
        hdr (numpy.ndarray):
            Header

    Returns:
        dict:
            Updated header values

    """
    lam = hdr["LAMBDA"] * 1000
    deltaLam = (hdr["MAXLAM"] - hdr["MINLAM"]) * 1000
    pupilDiam = hdr["D"]
    obscurFac = hdr["OBSCURED"]

    if header_vals is None:
        header_vals = {}

    inconsistent = False
    names = ["lam", "deltaLam", "pupilDiam", "obscurFac"]
    vals = [lam, deltaLam, pupilDiam, obscurFac]

    for n, v in zip(names, vals):
        if n not in header_vals:
            header_vals[n] = v
        else:
            if header_vals[n] != v:
                inconsistent = True

    if inconsistent:
        warnings.warn("Inconsistent header values in inputs.")

    return header_vals


def update_WA_vals(IWA, OWA, WAs):
    """Utility method for updating global IWA/OWA

    Args:
        IWA (float, optional):
            Current IWA value or None
        OWA (float, optional):
            Current OWA value or None
        WAs (numpy.ndarray):
            Angular separations of current system

    Returns:
        tuple:
            IWA (float or None)
            OWA (float or None)
    """

    if IWA is not None:
        if IWA < np.min(WAs):
            IWA = np.min(WAs)
    else:
        IWA = np.min(WAs)

    if OWA is not None:
        if OWA > np.max(WAs):
            OWA = np.max(WAs)
    else:
        OWA = np.max(WAs)

    return IWA, OWA
