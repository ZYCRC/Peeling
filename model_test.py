# %%
from xpbd_softbody import XPBDSoftbody
import pyvista as pv
import config as cfg
import matplotlib.pyplot as plt
import torch
from xpbd_softbody_layer import XPBDStep
import numpy as np
from sklearn.neighbors import KDTree
from xpbd_softbody_layer import get_energy_boundary
from scipy.interpolate import interp1d
from cubic_bezier import *
from tqdm import trange
from xpbd_softbody_layer_batch import XPBDStepBatch
from xpbd_softbody_layer_batch import get_energy_boundary_batch

# %%
control_point = 43

# %%
# load data
softbody = XPBDSoftbody()
# skin_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0), x_length=0.02, y_length=0.02, z_length=0.004), n_surf=N_skin)
# bandage_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0.003), x_length=0.02, y_length=0.02, z_length=0.002), n_surf=N_bandage)
# skin_mesh = softbody.add_mesh(pv.Cube(center=(0, 0, 0), x_length=0.02, y_length=0.02, z_length=0.004), n_surf=N_skin)
bandage_mesh = softbody.add_mesh(pv.Cube(center=(0, 0, 0.003), x_length=0.02, y_length=0.02, z_length=0.002), n_surf=400)
softbody.init_states()
softbody.init_dist_constraints()
# softbody.set_gravity(torch.tensor([0, 0, -9.8]).to(cfg.device))
# softbody.init_shape_constraints_thinshell([0, 1])
# softbody.add_multi_boundary_constrain(1, 0, 0.0017, contact_sur_0, contact_sur_1)
# softbody.set_gravity(torch.tensor([0, 0, -9.8]).to(cfg.device))
# softbody.fix_less_than(0, -0.001, 2)
softbody.fix_point(0, control_point)
# softbody.fix_less_than(1, 0.0022, 2)
# softbody.fix_larger_than(0, 0.0018, 2)

# %%
cloth_dist_stiffness = 1e10
V_boundary_stiffness_ref = 0
V_dist_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device) * cloth_dist_stiffness
# V_boundary_stiffness = torch.ones((softbody.C_boundary_list[0].shape[0], 1)).to(cfg.device) * V_boundary_stiffness_ref
# V_boundary_stiffness = V_boundary_stiffness.type(torch.DoubleTensor)
pl = pv.Plotter()
pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 2')
# pl.add_lines(spline_trajectory.detach().cpu().numpy(), connected=True, color='r')
# pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
pl.add_legend()
pl.camera_position = [(0.04943279151406822, -0.03977536222547223, 0.021820325700180415),
                        (7.301401243626354e-06, 2.0879731106636726e-05, 0.003000000142492353),
                        (-0.23015530645193596, 0.1675075959527941, 0.95862909418092)]
# pl.camera_position = 'xy'

# %%
for i in range(20):
    softbody.V[control_point][-1] += 0.005
    step_ref = step_ref = XPBDStep(softbody,
                V_dist_stiffness=V_dist_stiffness, 
                V_shape_stiffness=0,
                V_boundary_stiffness=0, 
                dt=cfg.dt,
                substep=cfg.substep,
                iteration=cfg.iteration,
                quasi_static=cfg.quasi_static,
                plane_height=cfg.ground_plane_height, 
                use_shape_matching=False,
                use_spring_boundary=False,
                use_dist=True)
    V_ref, V_velocity_ref = step_ref.forward(softbody.V, softbody.V_velocity)
    softbody.V = V_ref.clone()
    softbody.V_velocity = V_velocity_ref.clone()

    bandage_mesh.points = softbody.V.cpu().numpy()
    # pl.camera_position = 'xy'
    pl.show(interactive_update=True)

