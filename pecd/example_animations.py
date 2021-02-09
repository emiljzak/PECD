# Example animations using matplotlib's FuncAnimation
# Ken Hughes. August 2019

# For more detail, see
#https://brushingupscience.com/2019/08/01/elaborate-matplotlib-animations/

# Examples include
#   - quiver plot with variable positions and directions
#   - 3D contour plot
#   - line plot on a polar projection
#   - scatter plot with variable size, color, and shape
#   - filled area plot


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D

# Use matplotlib ggplot stylesheet if available
try:
    plt.style.use('ggplot')
except OSError:
    pass

# Set which type of animation will be plotted. One of:
# quiver, 3d_contour, polar, scatter, fill
animation_type = 'polar'

Nframes = 120

if animation_type == 'quiver':

    # Create an invisible axis
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_aspect('equal')
    ax.set_axis_off()
    xlim = 4
    ax.set(xlim=(-xlim, xlim), ylim=(-3, 3),
           xticklabels=[], yticklabels=[])

    x = np.linspace(0, 2*np.pi, 30, endpoint=False)
    t = np.linspace(0, 2*np.pi, Nframes, endpoint=False)
    # Arrays Qx and Qy are the positions of the base
    # points of the arrow with size (Nframes, Narrows) where
    # Narrows is the number of arrows
    Qx = np.sin(t)[:, None] + np.cos(x[None, :] + t[:, None])
    Qy = np.cos(t)[:, None] + np.cos(2*x[None, :] - t[:, None])
    # Arrow vectors are the same shape as Qx and Qy
    U = np.roll(Qy, 5) + np.sin(t[:, None])
    V = np.roll(Qx, 25)

    # Make an RGBA array with one entry for each frame
    # C is size (Nframes × 4)
    C = get_cmap('hsv', Nframes)(np.r_[:Nframes])
    C[:, -1] = 0.7

    # For frame 1, plot the 0th set of arrows
    s = np.s_[0, :]
    qax = ax.quiver(Qx[s], Qy[s], U[s], V[s],
                    facecolor=C[0], scale=10)
    fig.tight_layout()

    def animate(i):
        # Update to frame i
        s = np.s_[i, :]
        # Change direction of arrows
        qax.set_UVC(U[s], V[s])
        # Change base position of arrows
        qax.set_offsets(np.c_[Qx[s].flatten(), Qy[s].flatten()])
        # Change color of arrows
        Cidx = np.mod(np.r_[i:i+U.shape[1]], Nframes)
        # C[Cidx, :] is size (Narrows, 4)
        qax.set_facecolor(C[Cidx, :])


if animation_type == '3d_contour':
    # Create an invisible 3D axis
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig)
    ax.set_axis_off()
    ax.set_facecolor('white')
    ax.set_zlim([-1.5, 1.5])
    ax.set(xlim=(0, 2*np.pi), ylim=(0, 2*np.pi))

    # Create 2D grids of size (Ny, Nx)
    Nx, Ny = 30, 20
    x = np.linspace(0, 2*np.pi, Nx)
    y = np.linspace(0, 2*np.pi, Ny)
    X, Y = np.meshgrid(x, y)

    # Create wavy surface of size (Nframes, Ny, Nx)
    t = np.linspace(0, 2*np.pi, Nframes, endpoint=False)
    Z = (np.sin(x[None, None, :] + t[:, None, None]) +
         np.cos(y[None, :, None] + 2*t[:, None, None]))

    # Specify contour keyword options as a dict since
    # we'll need to pass this twice
    contour_opts = dict(
        vmin=-2, vmax=2, cmap=get_cmap('Blues'),
        edgecolors=None, alpha=1, antialiased=False)

    # Plot the 0th frame
    ax.plot_surface(X, Y, Z[0, :, :], **contour_opts)

    def animate(i):
        # Delete the existing contour
        ax.collections = []
        # Plot the ith contour
        ax.plot_surface(X, Y, Z[i, :, :], **contour_opts)


# ----------------------------------------------------------------------------
if animation_type == 'polar':
    # Create an axes with a polar projection
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_ylim([-2, 2])
    ax.set(yticks=np.r_[-1:3])

    # Create two similar curves
    r = np.linspace(0, 2*np.pi, 100)
    t = np.linspace(0, 2*np.pi, Nframes, endpoint=False)

    # theta1 and theta2 have size (Nr, Nframes) where
    # Nr is the number of elements making up each curve
    theta1 = (np.sin(r)[None, :]*np.cos(2*t)[:, None] +
              np.cos(2*r)[None, :]*np.cos(5*t)[:, None])
    theta2 = (np.sin(1*r)[None, :]*np.cos(t)[:, None] +
              np.cos(2*r)[None, :]*np.cos(3*t)[:, None])

    # Plot the 0th curves
    curve1 = ax.plot(r, theta1[0, :])[0]
    curve2 = ax.plot(r, theta2[0, :])[0]

    def animate(i):
        # Updating lines on a polar plot is no different
        # to updating lines on a regular plot
        curve1.set_ydata(theta1[i, :])
        curve2.set_ydata(theta2[i, :])


# ----------------------------------------------------------------------------
if animation_type == 'scatter':

    # Create an invisible axis
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off')

    # Create a grid of Nx × Ny scatter points
    Nx, Ny = 8, 8
    t = np.linspace(0, 2*np.pi, Nframes, endpoint=False)
    x, y = np.linspace(-3, 3, Nx), np.linspace(-3, 3, Ny)
    X, Y = np.meshgrid(x, y)
    # The locations of the scatter points change in time
    # X3 and Y3 have size (Nframes, Ny, Nx)
    X3 = X + np.exp(-X**2)[None, :, :]*np.sin(t)[:, None, None]
    Y3 = Y + np.exp(-Y**2)[None, :, :]*np.cos(t)[:, None, None]
    # Calculate the squared radius for convenience
    R = X3**2 + Y3**2
    # The size of the scatter points is also (Nframes, Ny, Nx)
    S = 15*(20-R)*(np.cos(2*t[:, None, None]+X3) +
                   np.sin(t[:, None, None]+Y3))
    S[S < 5] = 5

    # Each scatter point in each frame has a RGBA color, so
    # the color array C has a size (Nframes, Ny, Nx, 4)
    C = R/np.max(R) + np.sin(t[:, None, None])
    C = (C - np.min(C))/np.ptp(C)
    C = get_cmap('plasma', Nframes)(C)

    # Plot the 0th frame with color reshaped so that it
    # has the size (Ny × Ny, 4)
    scat = ax.scatter(X3[0, ...], Y3[0, ...], s=S[0, ...],
                      c=C[0, ...].reshape(-1, 4))

    def animate(i):
        # For the ith frame, pass the positions as an
        # array of size (Ny × Nx, 2).
        Xi, Yi = X3[i, :, :].flatten(), Y3[i, :, :].flatten()
        # np.c_[] helps combines the two 1D arrays appropriately
        scat.set_offsets(np.c_[Xi, Yi])
        # Sizes are passed as a 1D array
        scat.set_sizes(S[i, :, :].flatten())
        # Colors are passed as a array of size (Ny × Nx, 4)
        scat.set_facecolors(C[i, :, :].reshape(-1, 4))


# ----------------------------------------------------------------------------
if animation_type == 'fill':
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim([0, 2*np.pi])
    ax.set(xticks=np.linspace(0, 2*np.pi, 5),
           xticklabels=['0', 'π/2', 'π', '3π/2', '2π'])
    # ax.set_axis_off()

    Nx = 40
    Nt = Nframes
    x = np.linspace(0, 2*np.pi, Nx)
    t = np.linspace(0, 2*np.pi, Nt)
    # Create the upper and lower curves to fill between
    # Y1 and Y2 have size (Nx, Nframes) where Nx is the
    # number of elements in the curve
    Y1 = np.cos(t[:, None] - x[None, :])
    Y2 = 0.5*np.sin(2*t[:, None])*np.cos(x[None, :])

    C = get_cmap('coolwarm', Nframes)(np.r_[:Nframes:2, Nframes:0:-2]/Nframes)

    # Plot the 0th frame
    fill = ax.fill_between(x, Y1[0, :], Y2[0, :])

    def animate(i):
        # Get the paths created by fill_between
        path = fill.get_paths()[0]
        # vertices is the part we need to change
        verts = path.vertices
        # Elements 1:Nx+1 encode the upper curve
        verts[1:Nx+1, 1] = Y1[i, :]
        # Elements Nx+2:-1 encode the lower curve, but
        # in right-to-left order, hence the need
        # to specify [::-1] to reverse the curve
        verts[Nx+2:-1, 1] = Y2[i, :][::-1]
        # It is unclear what 0th, Nx+1, and -1 elements
        # are for as they are not addressed here

        # Change the color just because we can
        fill.set_color(C[i, :])


# ----------------------------------------------------------------------------
# Save the animation
anim = FuncAnimation(
    fig, animate, interval=50, frames=Nframes, repeat=True)
fig.show()
# anim.save(animation_type + '.gif', writer='imagemagick')