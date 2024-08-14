import torch
import matplotlib as mplt
import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Stiffness related parameters 
mesh_bound = 1e3
conn_bound = 0.2

# PBD Simulation Related Parameters
dt = 0.01
substep = 10
iteration = 1

use_spring_boundary = True 
use_shape_matching = False 
quasi_static = True 

ground_plane_height = torch.tensor([-torch.inf]).to(device)

n_surf = 600

default_V_dist_stiffness = 1e10
default_V_boundary_stiffness = 1e-4

init_sigma_val = 0.5

large_step_iteration = 70

