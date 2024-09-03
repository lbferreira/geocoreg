import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray


def export_timeseries_animation(
    ts: xarray.DataArray,
    time_dim: str,
    filename: str = "animation.gif",
    fps: int = 3,
):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initialize the plot with the first time step
    ts.isel({time_dim: 0}).plot.imshow(ax=ax, cmap="viridis", robust=True, add_colorbar=False)

    def update(frame):
        ax.clear()
        ts.isel({time_dim: frame}).plot.imshow(
            ax=ax, cmap="viridis", robust=True, add_colorbar=False
        )
        ax.set_title(f"Time step: {frame}")

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(ts[time_dim]), repeat=True)

    ani.save(filename, fps=fps, dpi=200)
    plt.close()
    print(f"Animation saved to {filename}")
