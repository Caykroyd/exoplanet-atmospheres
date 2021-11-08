from matplotlib import pyplot as plt
import numpy as np

from operator import itemgetter
from core.vector import ray_intersects_sphere
from core.vector     import Vector3

def set_axes_equal(ax):
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    # Source:
    # https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_arrow(ax, *args, **kwargs):
    a = Arrow3D(*args,**kwargs)
    ax.add_artist(a)


def plot_sphere(ax, position, radius):
    x0, y0, z0 = position
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = radius*np.cos(u)*np.sin(v)
    y = radius*np.sin(u)*np.sin(v)
    z = radius*np.cos(v)
    ax.plot_surface(x+x0, y+y0, z+z0, color="r", alpha=0.3)


def ostream(scene, time, dt):
    cam = scene.objs['cam']
    star = scene.objs['star']
    out = {
        'time': time,
        'cam': cam.transform.position,
        'trajectory': cam.transform.global_to_local_coords(star.transform.position)
    }
    return out

def plot_axes(ax,center, e_x, e_y, e_z, length, linewidth=2, colors=['r', 'g', 'b'], **kwargs):
    c1, c2, c3 = colors
    plot_arrow(ax, *zip(center, center+length*e_x), mutation_scale=10, lw=linewidth, color=c1, **kwargs)
    plot_arrow(ax, *zip(center, center+length*e_y), mutation_scale=10, lw=linewidth, color=c2, **kwargs)
    plot_arrow(ax, *zip(center, center+length*e_z), mutation_scale=10, lw=linewidth, color=c3, **kwargs)
    #
    # ax.plot(*zip(center, center+length*e_x), linewidth=linewidth, **kwargs)
    # ax.plot(*zip(center, center+length*e_y), linewidth=linewidth, **kwargs)
    # ax.plot(*zip(center, center+length*e_z), linewidth=linewidth, **kwargs)

def plot_setup(fig, ax, scene):
    '''
    Plots the Planet-Observer setup
    '''
    cam, planet, star = itemgetter('cam', 'planet', 'star')(scene.objs)

    scene.set_time(0)
    data = scene.update(duration = 24*3600, dt = 60, ostream=ostream)

    r = planet.radius

    plot_sphere(ax, planet.transform.position, planet.radius)

    # Plot origin
    ax.scatter3D(*planet.transform.position, marker='+')

    # Plot tilt axis
    tilt_axis = planet.transform.local_to_global_coords(Vector3(0,0,1))
    ax.plot(*zip(-tilt_axis*r, tilt_axis*r), linewidth=3)

    center = cam.transform.position
    e_x = cam.transform.local_to_global_coords(Vector3(1,0,0))
    e_y = cam.transform.local_to_global_coords(Vector3(0,1,0))
    e_z = cam.transform.local_to_global_coords(Vector3(0,0,1))

    plot_axes(ax, center, e_x-center, e_y-center, e_z-center, r/2, linewidth=3, arrowstyle="-")

    # TODO: Generer ces donnes a la main: creer un cercle de rayon bla et blah
    X, Y, Z = zip(*data['cam'])
    ax.plot(X, Y, Z, linewidth=2, linestyle='dashed', c='k')

    center = planet.transform.position
    e_s = (star.transform.position-planet.transform.position).normalize()
    axis_length = 3*r

    plot_axes(ax,center, Vector3(1,0,0), Vector3(0,1,0), Vector3(0,0,1), axis_length, linewidth=1, arrowstyle="-|>")
    # plot_arrow(ax, *zip(center, center+axis_length*e_s), mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.text(*(center+axis_length*e_s), 'Star', e_s, ha='left', va='bottom')

    ax.set_title('Planet-Observer Setup')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.scatter([-axis_length,axis_length],[-axis_length,axis_length],[-axis_length,axis_length], alpha=0)
    #set_axes_equal(ax))
    # fig.subplots_adjust(left=-0.2, right=1.2, top=1, bottom=0)
    ax.grid(False)
    ax.axis('off')
    fig.tight_layout()

def plot_star_trajectory(cam, time, points):

    X, Y, Z = np.array([*zip(*points)])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Trajectory of the star in observer\'s frame')
    ax.scatter3D(0,0,0,marker='+')
    ax.plot(X, Y, Z, linewidth=3)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    set_axes_equal(plt.gca())
    ax.grid(False)
    ax.axis('off')


def plot_star_trajectory_projections(cam, time, points):

    X, Y, Z = np.array([*zip(*points)])

    plt.figure(figsize=(15,4))

    plt.subplot(131)
    plt.plot(time, X, linewidth=2)
    plt.ylabel('X [m]')
    plt.xlabel('time [s]')
    plt.subplot(132)
    plt.plot(time, Y, linewidth=2)
    plt.ylabel('Y [m]')
    plt.xlabel('time [s]')
    plt.subplot(133)
    plt.plot(time, Z, linewidth=2)
    plt.ylabel('Z [m]')
    plt.xlabel('time [s]')
    plt.subplots_adjust(0.05,0.15,0.95,0.9)
    plt.suptitle('Trajectory of the star in observer\'s frame')

def plot_star_trajectory_on_canvas(fig, ax, scene):

    cam = scene.objs['cam']

    scene.set_time(0)
    data = scene.update(duration = 24*3600, dt = 60, ostream=ostream)

    time, points = data['time'], data['trajectory']

    X, Y, Z = np.array([*zip(*points)])

    X, Y, indices = cam.to_canvas((X,Y,Z))

    order = np.argsort(X)
    X, Y = X[order], Y[order]
    # def consecutive(data):
    #     return np.split(data, np.where(np.diff(data) != 1)+1)
    # traj = consecutive(indices)

    ax.set_title('Trajectory of the star as seen by observer')
    ax.plot(X, Y, linewidth=2, linestyle='dotted', color='k')

    N = len(X)
    dN = len(X) // 3
    for n in range(dN, N-1, dN+1):
        ax.arrow(X[n], Y[n], X[n]-X[n-dN//8], Y[n]-Y[n-dN//8], head_width=3, color='k')

    lim = cam.canvas_width/2

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')

def plot_spectrum(fig, ax, scene, range, res=10**3):
    cam, star = itemgetter('cam', 'star')(scene.objs)

    # ray = axis #cam.transform.local_to_global_coords(axis)
    hit = True #ray_intersects_sphere(cam.transform.position, ray, star.transform.position, star.radius)

    freq = np.geomspace(*range, res)

    if hit:
        B = star.spectrum(freq)
    else:
        B = np.zeros_like(freq)
        print('Didnt intersect')

    # plt.title(r'Spectrum along axis {}'.format(tuple(axis)))
    ax.set_title('Spectrum along star-planet axis')
    ax.plot(freq, B, linestyle='dotted', c='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\nu \quad [Hz]$')
    ax.set_ylabel(r'$I_\nu \quad [W \: sr^{-1} \: m^{-2} \: Hz^{-1}]$')

    freq_min, freq_max = cam.frequency_band
    band = (freq >= freq_min) & (freq <= freq_max)
    ax.fill_between(freq, B, where=band, color='r')

def fluxmap_plotter():
    setup = True # define the closure

    def plot_fluxmap(fig, ax, scene):
        cam, planet, star = itemgetter('cam', 'planet', 'star')(scene.objs)

        def intensity(x, y, z):
            # B = cam.band_integrate(star.spectrum)
            I_R,I_G,I_B = cam.RGB(star.spectrum)
            d = star.transform.position - cam.transform.position

            mask = (np.sqrt(y**2 + z**2) * d.norm() <  x * star.radius)
            R = np.where(mask, I_R, 0)
            G = np.where(mask, I_G, 0)
            B = np.where(mask, I_B, 0)

            return np.stack([R,G,B])

        F = cam.capture(intensity)

        F = np.moveaxis(F,0,-1)

        ax.set_title(r'Flux seen by camera [$W \: m^{-2}$]')
        lim = cam.canvas_width/2
        cmap = ax.imshow(F,extent=[-lim,lim,-lim,lim], cmap='gray',vmin=0, origin='lower')
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')

        nonlocal setup
        if setup:
            fig.colorbar(cmap)
            setup = False

    return plot_fluxmap
