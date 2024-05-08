import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cri.transforms

force = [1.3118, 1.2013, 0.9949, 0.0000]
pose = [19.9049,  6.3413,  7.6324, -9.2554,  7.4815, 16.2354,  0.0000,  0.0000]
finger_name = ['index', 'thumb', 'middle', 'pinky']
colors = ['r', 'b', 'y', 'g']

def plot_contact_patch(ax, x, y, z, sphere_radius, patch_radius, theta_center, azimuth_center, color):

    # Calculate the cartesian coordinates of the patch center
    x_center = sphere_radius * np.sin(theta_center) * np.cos(azimuth_center)
    y_center = sphere_radius * np.sin(theta_center) * np.sin(azimuth_center)
    z_center = sphere_radius * np.cos(theta_center)

    distances = np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2)
    patch = distances <= patch_radius

    # Check which points are within the patch
    distances = np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2)
    patch = distances <= patch_radius

    # Highlight the patch
    scatter = ax.scatter(x[patch], y[patch], z[patch], color=color, alpha=0.01)
    return scatter


def set_3d_axes_equal(ax):
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

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
graph,  = ax.plot([], [], [], linestyle="", marker="o", color='b')
quiver_fz = ax.quiver(0, 0, 0, 0, 0, 0)

# plot hemisphere wireframe
tip_rad = 11.2
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:0.5*pi:180j,
                        0.0:2.0*pi:720j]  # phi = alti, theta = azi
x_sphere = tip_rad*sin(phi)*cos(theta)
y_sphere = tip_rad*sin(phi)*sin(theta)
z_sphere = tip_rad*cos(phi)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="k", linewidth=0.2)
set_3d_axes_equal(ax)
ax.grid(False)
ax.set_axis_off()

# plt.xlabel('Contact Pose Y')
# plt.ylabel('Contact Pose X')

# Plot force arrow
for i in range(4):
    point_offset = 0.0
    point = np.array([0.0, 0.0, tip_rad + point_offset])
    roll = pose[i*2 + 1] * 1.5
    pitch = pose[i*2]*1.5
    fz_scale = force[i] * 5

    # Calculate contact point
    rot_mat = cri.transforms.euler2mat(
        [0.0, 0.0, 0.0, roll, pitch, 0.0])[:3, :3]
    x, y, z = np.dot(point, rot_mat)
    fzu, fzv, fzw = np.dot(np.array([0.0, 0.0, 1.0]), rot_mat)

    # Plot contact point
    graph.set_data(x, y)
    graph.set_3d_properties(z)
    quiver_fz.remove()
    quiver_fz = ax.quiver(x, y, z, fzu, fzv, fzw,
                            length=-fz_scale, normalize=True, color='b')
    
    # PLot contact region
    phi0 = np.arctan2(y, x)  # Azimuthal angle
    theta0 = np.arccos(z / tip_rad)  # Polar angle
    scatter = plot_contact_patch(ax,  
                                 x_sphere, 
                                 y_sphere, 
                                 z_sphere,
                                sphere_radius=tip_rad, 
                                patch_radius=fz_scale/2, 
                                theta_center=theta0, 
                                azimuth_center=phi0,
                                color=colors[i])

    plt.tick_params(left = True, right = True , labelleft = False , 
                    labelbottom = False, bottom = False) 
    

    plt.tight_layout()

    # plt.show()

    root_directory = '/home/max-yang/Documents/Projects/allegro/smg_gym/smg_gym/ppo_adapt_helpers/analysis/data/'
    save_image_path = os.path.join(root_directory, f'contact_{finger_name[i]}.png')

    plt.savefig(save_image_path)
    scatter.remove()