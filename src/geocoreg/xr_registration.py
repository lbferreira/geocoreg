from typing import Dict, List, Optional, Union
import xarray
from . import registrators, registrator_factory


def coregistrate_images():
    # TODO: Implement this function to coregistrate a single pair of images.
    pass


def coregistrate_timeseries(
    scr_timeseries: xarray.DataArray,
    dst_img: xarray.DataArray,
    registrator: Union[str, registrators.Registrator] = "pcc",
    registration_bands: Optional[List[str]] = None,
    x_dim: str = "x",
    y_dim: str = "y",
    time_dim: str = "time",
    band_dim: str = "band",
    chunks: Optional[Dict[str, int]] = None,
) -> xarray.DataArray:
    """Coregistrate a time series of images to a reference image.

    Args:
        scr_timeseries (xarray.DataArray): image time series to be coregistrated.
        dst_img (xarray.DataArray): image to be used as reference.
        registrator (Union[str, registrators.Registrator], optional): a str defining the registrator to be used or a registrators.Registrator object. Check registrator_factory.get_available_registrators() to see valid string values. Defaults to "pcc".
        registration_bands (Optional[List[str]], optional): bands to be used in the registration. Defaults to None.
        x_dim (str, optional): name of the x dimension. Defaults to "x".
        y_dim (str, optional): name of the y dimension. Defaults to "y".
        time_dim (str, optional): name of the time dimension. Defaults to "time".
        band_dim (str, optional): name of the band dimension. Defaults to "band".
        chunks (Optional[Dict[str, int]], optional): chunks to be used in the resulting DataArray. Defaults to None.

    Returns:
        xarray.DataArray: Coregistrated timeseries.
    """
    assert set(scr_timeseries.dims) == {
        x_dim,
        y_dim,
        time_dim,
        band_dim,
    }, "scr_timeseries must have dimensions x, y, time and band."
    assert set(dst_img.dims) == {
        x_dim,
        y_dim,
        band_dim,
    }, "dst_img must have only the dimensions x, y and band."

    if registration_bands is None:
        registration_bands = dst_img[band_dim].to_numpy().tolist()

    if isinstance(registrator, str):
        registrator = registrator_factory.build_registrator(registrator)

    dst_img_registration_np = (
        dst_img.sel({band_dim: registration_bands}).transpose(y_dim, x_dim, band_dim).to_numpy()
    )
    scr_timeseries_registered = scr_timeseries.transpose(time_dim, y_dim, x_dim, band_dim).copy()
    for i in range(scr_timeseries[time_dim].shape[0]):
        scr_img = scr_timeseries.isel({time_dim: i}).transpose(y_dim, x_dim, band_dim)
        scr_img_registration_np = scr_img.sel({band_dim: registration_bands}).to_numpy()
        scr_img_np = scr_img.to_numpy()

        registrator.register(scr_img_registration_np, dst_img_registration_np)
        scr_img_np_registered = registrator.warp_image(scr_img_np)
        # The first axis is the time axis since the image was transposed to have the time axis as the first axis
        scr_timeseries_registered.data[i, ...] = scr_img_np_registered
        if chunks is not None:
            scr_timeseries_registered = scr_timeseries_registered.chunk(chunks)

    return scr_timeseries_registered
