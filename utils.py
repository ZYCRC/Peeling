import numpy as np
import torch
import os
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
import pyvista as pv
import config as cfg
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
class Visualizer:
    def __init__(self):
        pv.set_plot_theme('document')
        self.pl= pv.Plotter()
        self.pl.add_axes()


    def reset(self, mesh_map):
        self.meshes = {}
        self.mesh_info = {}

        for key in mesh_map:
            mesh, style, scalars, clim = mesh_map[key]
            self.meshes[key] = mesh
            self.mesh_info[key] = [style, scalars, clim]

        self.reset_visualizer()

    def reset_visualizer(self):
        self.pl.clear()
        for key in self.meshes:
            mesh = self.meshes[key]
            style, scalars, clim = self.mesh_info[key]

            if scalars is None:
                scalars = np.ones(mesh.points.shape[0])
            opacity = 0.1 if style == 'points' else 1
            self.pl.add_mesh(mesh,
                             show_edges=True,
                             lighting=True,
                             opacity=opacity,
                             scalars=scalars,
                             style=style,
                             cmap='viridis', # config colormaps
                             clim=clim)
        self.pl.add_scalar_bar()
        self.pl.show(interactive_update=True)


        self.pl.camera.position = np.array([0.07, -0.00, 0.2])



    def update(self, update_map, new_clim=None, save=False, name=None, save_path=None):
        for key in update_map:
            points, colors = update_map[key]
            self.meshes[key].points = points
            if colors is not None:
                self.meshes[key].point_data.set_scalars(colors, name='Colors')
                self.mesh_info[key][1] = colors
            if new_clim is not None:
                self.mesh_info[key][2] = new_clim

        if new_clim is not None:
            self.reset_visualizer()

        self.pl.render()
        if save:
            self.pl.save_graphic(os.path.join(save_path, name + '.svg'))

    def close(self):
        self.pl.close()

    def calibrate_camera(self):
        '''
        Utility function to easily print camera parameters
        '''
        # get camera parameter
        focal = self.pl.camera.focal_point
        pos = self.pl.camera.position
        up = self.pl.camera.up
        print(f'Focal Point: ({focal[0]:.3f}, {focal[1]:.3f}, {focal[2]:.3f})')
        print(f'Camera Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')
        print(f'Camera Up: ({up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f})')
        # update visualizer
        self.pl.render()
        self.pl.iren.process_events()



def convert_stiffness(stiffness, max=10):
    return F.sigmoid(stiffness) * max  + 1e-10


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def convert_to_normalize_coordinate(coors, range=None):
    if range is None:
        init_x_min, init_x_max, init_y_min, init_y_max = coors[:, 0].min(), coors[:, 0].max(), coors[:, 1].min(), coors[:, 1].max()
    else:
        init_y_min, init_y_max, init_x_min, init_x_max = range

    pad_x_min = init_x_min
    pad_x_max = init_x_max
    pad_y_min = init_y_min
    pad_y_max = init_y_max
    new_coors = coors.clone()
    new_coors[..., 0] -= pad_x_min
    new_coors[..., 0] /= pad_x_max - pad_x_min
    new_coors[..., 1] -= pad_y_min
    new_coors[..., 1] /= pad_y_max - pad_y_min
    return new_coors

def field_from_2d_points(bottom_points, range=None):
    if range is None:
        x_range, y_range = 500, 500
    else:
        _, x_range, _, y_range = range
    grid_x, grid_y = np.mgrid[0:1:y_range*1j, 0:1:x_range*1j]
    norm_coors = convert_to_normalize_coordinate(bottom_points, range)
    points, value = norm_coors[:, :-1], norm_coors[:, -1:]
    grid = griddata(points.cpu().numpy(), value.cpu().numpy(), (grid_x, grid_y), method='linear', fill_value=0)
    # grid = ndimage.gaussian_filter(grid, sigma=(7, 7, 0), order=0)
    return grid