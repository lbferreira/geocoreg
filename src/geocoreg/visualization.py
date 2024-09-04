import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray


def export_animation(
    image_series: xarray.DataArray,
    variation_dim: str,
    file_name: str,
    fps: int = 3,
) -> None:
    """Export an animation of a image series. It's useful to visualize image misalignments.

    Args:
        image_series (xarray.DataArray): image series to be exported as animation.
        variation_dim (str): name of the dimension in the image series that will be used as variation in the animation.
        file_name (str): path to the file where the animation will be saved.
        fps (int, optional): frames per second. Defaults to 3.
    """
    fig, ax = plt.subplots()

    # Initialize the plot with the first step
    image_series.isel({variation_dim: 0}).plot.imshow(
        ax=ax, cmap="viridis", robust=True, add_colorbar=False
    )

    def update(frame):
        ax.clear()
        image_series.isel({variation_dim: frame}).plot.imshow(
            ax=ax, cmap="viridis", robust=True, add_colorbar=False
        )
        ax.set_title(f"Step: {frame}")

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(image_series[variation_dim]), repeat=True)
    ani.save(file_name, fps=fps, dpi=200)
    plt.close()
