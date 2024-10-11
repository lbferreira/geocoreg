from typing import Dict, List, Optional, Union
import numpy as np
import xarray
from . import registrators, registrator_factory


def coregistrate(
    scr_imgs: xarray.DataArray,
    dst_img: xarray.DataArray,
    registrator: Union[str, registrators.Registrator] = "pcc",
    registration_band: Union[str, List[str], None] = None,
    x_dim: str = "x",
    y_dim: str = "y",
    band_dim: str = "band",
    chunks: Optional[Dict[str, int]] = None,
) -> xarray.DataArray:
    """Coregistrate an image of a set of images to a reference image.
    All other dimensions, except x, y, and band, will be iterated over and the coregistration
    will be applied to each image (composed by x, y coordinates and a set of bands).
    This function can be used, for example, to coregistrate a time series of images to a reference image.

    Args:
        scr_imgs (xarray.DataArray): images (a single one or multiple) to be coregistrated in relation to the reference image.
        dst_img (xarray.DataArray): image to be used as reference.
        registrator (Union[str, registrators.Registrator], optional): a str defining the registrator to be used or a registrators.Registrator object. Check registrator_factory.get_available_registrators() to see valid string values. Defaults to "pcc".
        registration_band (Union[str, List[str], None], optional): band(s) to be used in the registration. Defaults to None.
        x_dim (str, optional): name of the x dimension. Defaults to "x".
        y_dim (str, optional): name of the y dimension. Defaults to "y".
        band_dim (str, optional): name of the band dimension. Defaults to "band".
        chunks (Optional[Dict[str, int]], optional): chunks to be used in the resulting DataArray. Defaults to None.

    Returns:
        xarray.DataArray: Coregistrated timeseries.
    """
    # TODO: check if it is really necessary to force the input to be a float dtype
    assert np.issubdtype(scr_imgs.dtype, np.floating), "scr_imgs must have a floating point dtype."
    assert (
        x_dim in scr_imgs.dims and y_dim in scr_imgs.dims and band_dim in scr_imgs.dims
    ), "scr_imgs must have at least the dimensions x, y, and band."
    assert set(dst_img.dims) == {
        x_dim,
        y_dim,
        band_dim,
    }, "dst_img must have only the dimensions x, y and band."

    if registration_band is None:
        registration_band = dst_img[band_dim].to_numpy().tolist()
    elif isinstance(registration_band, str):
        registration_band = [registration_band]

    if isinstance(registrator, str):
        registrator = registrator_factory.build_registrator(registrator)

    dst_img_registration_np = (
        dst_img.sel({band_dim: registration_band}).transpose(y_dim, x_dim, band_dim).to_numpy()
    )

    remaining_dims = set(scr_imgs.dims) - {x_dim, y_dim, band_dim}
    scr_images_registered = scr_imgs.transpose(*remaining_dims, y_dim, x_dim, band_dim).copy()

    def generate_registrated_image(scr_img):
        scr_img = scr_img.transpose(y_dim, x_dim, band_dim)
        scr_img_registration_np = scr_img.sel({band_dim: registration_band}).to_numpy()
        scr_img_np = scr_img.to_numpy()
        registrator.register(scr_img_registration_np, dst_img_registration_np)
        scr_img_np_registered = registrator.warp_image(scr_img_np)
        return scr_img_np_registered

    if len(remaining_dims) > 0:
        len_remaining_dims = [scr_imgs.sizes[dim] for dim in remaining_dims]
        idx_dims = [np.arange(dim_len) for dim_len in len_remaining_dims]
        for idx in np.ndindex(*len_remaining_dims):
            coord_dict = {dim: idx_dims[i][idx[i]] for i, dim in enumerate(remaining_dims)}
            scr_img = scr_imgs.isel(**coord_dict)
            scr_img_np_registered = generate_registrated_image(scr_img)
            scr_images_registered.data[(*idx, ...)] = scr_img_np_registered
            if chunks is not None:
                scr_images_registered = scr_images_registered.chunk(chunks)
    else:
        scr_img_np_registered = generate_registrated_image(scr_imgs)
        scr_images_registered.data[...] = scr_img_np_registered
        if chunks is not None:
            scr_images_registered = scr_images_registered.chunk(chunks)

    # Transpose back to the original dimension order
    scr_images_registered = scr_images_registered.transpose(*scr_imgs.dims)
    return scr_images_registered
