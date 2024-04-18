import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import argparse

import os
direc = '2d_energy/'
if not os.path.exists(direc):
    os.makedirs(direc)

parser = argparse.ArgumentParser('2D Flow Hamiltonian Demo')
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

ts = np.linspace(0, 5, 1000)
z_cache = None # Cached forward state. I've found backward solve of ODE f(z) blows up and doesn't work. Easier to cache forward pass.

def f(z, t):
    # z is [2, ...]. 
    return np.stack([2.5 - z[1] - (z[0] - 2.5)**3, z[0] - z[1]])

def J_z(z, t):
    return np.array([[-3. * (z[0] - 2.5)**2, -1.], [1., -1.]])

def f_a(a, z, t):
    # a and z are [2, ...].
    return -J_z(z, t).T.dot(a)

def f_aug(aug, t):
    # aug is [4, ...] and has aug[:2] = z, aug[2:] = a. 
    z, a = aug[:2], aug[2:]
    t_lerp = int(round(t * (len(ts)-1) / ts[-1]))
    z = z_cache[:, t_lerp] # USE CACHED FORWARD STATE.
    rex = np.concatenate([f(z, t), f_a(a, z, t)], 0)
    return rex

def L(z):
    # z is [2, ...].
    return np.sum(z**2) / 2.

def L_prime(z):
    # z is [2, ...].
    return z 

def H(z, a):
    # z and a are [2, ...].
    return (z * a).sum(0)

def plot_trajectory(ts, traj, name, cmap = mpl.colormaps['cool'], include_cbar = True):
    # traj is [2, tsteps]. Just a helper function for plotting trajectory with coloring based on time.
    nstep = traj.shape[1] // 10
    main_ax = plt.gca()
    for s in range(0, traj.shape[1], nstep):
        label = '_nolegend_' if s > 0 else name 
        main_ax.plot(traj[0, s:s+nstep+1], traj[1, s:s+nstep+1], color = cmap(s / float(traj.shape[1] - 1)), label = label, marker = '.')
    main_ax.set_facecolor("black")

    if include_cbar:
        ax, _ = mpl.colorbar.make_axes(plt.gca())
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=mpl.colors.Normalize(vmin=ts[0], vmax=ts[-1]))
        cb.set_label('Time (s)')
    return main_ax


def analyze(z0):
    global z_cache
    traj = odeint(f, z0, ts).T
    vel = f(traj, ts)
    z_cache = traj

    LT = L_prime(traj[:, -1])
    aug = np.concatenate([traj[:, -1], LT], 0)
    aug_traj = odeint(f_aug, aug, ts[::-1]).T
    a_traj = aug_traj[2:, ::-1]

    plt.figure(figsize = (8, 6))
    plt.subplot(2,2,1)
    ax = plot_trajectory(ts, traj, '$z(t)$', include_cbar=False)
    ax.scatter(2.5, 2.5, label = '$z_{\\infty}$', c = 'red', marker = 'x')
    ax.legend()

    plt.subplot(2,2,2)
    ax = plot_trajectory(ts, f(traj, ts), '$f(z(t))$', cmap = mpl.colormaps['autumn'], include_cbar=False)
    ax = plot_trajectory(ts, a_traj, '$a(t)$')
    ax.legend()

    plt.subplot(2,2,(3,4))
    plt.plot(ts, H(f(traj, ts), a_traj), '--', c = 'black')
    plt.title('$H(z(t), a(t))$')
    plt.savefig(direc + 'Figure_1.png')

    plt.figure()
    plt.plot(ts, a_traj[0], '-')
    plt.plot(ts, a_traj[1], '-')
    plt.plot(ts, vel[0], '-')
    plt.plot(ts, vel[1], '-')
    plt.plot(ts, H(f(traj, ts), a_traj), '--', c = 'black')
    plt.legend(['$a_1(t)$', '$a_2(t)$', '$f(z_1(t))$', '$f(z_2(t))$', '$H(z(t), a(t))$'])
    plt.savefig(direc + 'Figure_2.png')

    # 3D quiver plot. 
    for idx, normalize in enumerate([True, False]):
        s = 6 # Plot stride
        ax = plt.figure().add_subplot(projection='3d')
        ox, oy, oz = np.meshgrid(np.zeros(1),
                              np.zeros(1),
                              ts[::s])

        vx = vel[0, ::s].reshape(1, 1, -1)
        vy = vel[1, ::s].reshape(1, 1, -1)
        vz = np.zeros_like(oz)
        length = 0.07 if normalize else 0.02
        quiv = ax.quiver(ox, oy, oz, vx, vy, vz, cmap = 'autumn', length=length, normalize = normalize)
        quiv.set_array(ts[::s//3])

        vx = a_traj[0, ::s].reshape(1, 1, -1)
        vy = a_traj[1, ::s].reshape(1, 1, -1)
        quiv = ax.quiver(ox, oy, oz, vx, vy, vz, cmap='cool', length=length, normalize = normalize)
        quiv.set_array(ts[::s//3])

        plt.legend(['blue, $a(t)$', 'red, $f(z(t))$'])
        ax.set_zlabel('Time (s)')
        nm_str = 'Normalized' if normalize else 'Not Normalized'
        plt.title(f'Forward Backward Vectors Over Time ({nm_str})')
        plt.savefig(direc + f'Figure_{3+idx}.png')
    plt.show()


if __name__ == '__main__':
    analyze(z0 = np.array([-1, -2]))
