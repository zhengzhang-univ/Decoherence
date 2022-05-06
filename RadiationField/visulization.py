import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def animation_3D_surface_plot(x_train, y_train, zarray, name):
    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(x, y, zarray[:, :, frame_number], cmap="magma")

    nx, ny, nz = zarray.shape
    fps = 10  # frame per sec
    assert nx == len(x_train) and ny == len(y_train)
    x, y = np.meshgrid(x_train, y_train)

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    fig.set_dpi(200)
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(x, y, zarray[:, :, 0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(0, 1.01 * np.max(zarray))
    ani = animation.FuncAnimation(fig, update_plot, nz, fargs=(zarray, plot), interval=1000 / fps)
    ani.save(name + '.mp4', writer='ffmpeg', fps=fps)
    pass


def plots_2d_in_3d(tlist, array_2d):
    """
    "array_2d" has the shape of (times,observables).
    """
    ts, ns = array_2d.shape
    assert ts == len(tlist)
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    ax = fig.add_subplot(projection='3d')
    for i in range(ns):
        ax.plot(tlist, array_2d[:,i], zs=i, zdir='y')

    #ax.legend()
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    #ax.set_zlim(0, 1)
    ax.set_xlabel('time')
    ax.set_ylabel('$\chi$')
    #ax.set_zlabel('$<N>$')

    ax.view_init(elev=20., azim=-35)

    plt.show()
    pass
"""
def animation_imshow(array3d):
    fig = plt.figure()
    time_ind = 0
    im = plt.imshow(array3d[time_ind], animated=True)

    def updatefig(*args):
        global time_ind
        time_ind+=1
        im.set_array(array3d[time_ind])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()
    pass
"""

def animation_imshow(array3d):
    fig, ax = plt.subplots()
    n1,n2,n3 = array3d.shape
    ims = []
    for i in range(n1):
        im = ax.imshow(array3d[i,:,:], animated=True)
        if i == 0:
            ax.imshow(array3d[i,:,:])  # show an initial one first
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    # To save the animation, use e.g.
    #
    ani.save("Desmovie.mp4")
    #
    # or
    #
    #writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save("/Users/zheng/Desktop/movie.mp4", writer=writer)

    plt.show()