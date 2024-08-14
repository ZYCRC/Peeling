import os
import numpy as np
import pyvista as pv
import torch
import cv2
import pyacvd
import json
import transforms3d as t3d
from torch.utils.data import Dataset
from xpbd_softbody import XPBDSoftbody
from xpbd_softbody_layer import XPBDStep
import config as cfg

def get_xpbd_grape(path='../assets'):
    softbody = XPBDSoftbody()
    mesh = softbody.add_thinshell(pv.read(os.path.join(path, 'grape_skin.ply')), n_surf=cfg.n_surf)
    # mesh = softbody.add_thinshell(pv.Plane(), n_surf=cfg.n_surf)
    softbody.init_states()
    print(softbody.V.shape)
    softbody.init_spring_boundary()
    print(softbody.V.shape)
    softbody.init_boundary_constraints()
    softbody.init_dist_constraints()
    softbody.set_gravity(torch.tensor([0, 0, -9.8]).to(cfg.device))
    softbody.fix_virtual_boundary()
    # NOTE define a contact plane
    # softbody.define_contact_field(softbody.V)
    return mesh, softbody

