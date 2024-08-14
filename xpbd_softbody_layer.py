import torch
from typing import Tuple
from xpbd_softbody import XPBDSoftbody
import utils
import config as cfg



class XPBDStep(torch.nn.Module):
    def __init__(self, softbody: XPBDSoftbody,
                 V_dist_stiffness: torch.tensor,
                 V_shape_stiffness: torch.tensor,
                 V_boundary_stiffness: torch.tensor,
                 dt: float = 0.01,
                 substep: int = 1,
                 iteration: int = 10,
                 plane_height=-torch.inf,
                 quasi_static=False,
                 use_shape_matching=False,
                 use_spring_boundary=False,
                 use_dist=False, 
                 using_functorch=False) -> None:
        '''
        Initialize step function

        Args:
            softbody (XPBDSotbody): softbody object
            dt (float): step size
            substep: substep count
            iteration (int): solver iteration count
            dist_stiffness (float): distance stiffness
            vol_stiffness (float): volume stiffness
            td_stiffness (float): tendon stiffness
        '''
        super().__init__()
        # state tensors
        self.V_mass = softbody.V_mass
        self.V_force = softbody.V_force
        self.V_w = softbody.V_w
        self.softbody = softbody
        # solver parameters
        self.dt = dt
        self.substep = substep
        self.iteration = iteration
        self.plane_height = plane_height
        # constraint projection layers
        self.project_list = []
        self.L_list = []
        self.quasi_static = quasi_static

        self.using_functorch = using_functorch

        # distance constraints
        if use_dist:
            V_dist_compliance = 1 / (V_dist_stiffness * (dt / substep)**2)
            for C_dist, C_init_d in zip(softbody.C_dist_list, softbody.C_init_d_list):
                # print(C_init_d.shape)
                self.L_list.append(torch.zeros_like(C_init_d).to(cfg.device))
                self.project_list.append(project_C_dist(
                    softbody.V_w, V_dist_compliance, C_dist, C_init_d
                ))

        # shape matching constraints
        if use_shape_matching:
            V_shape_compliance = 1 / (V_shape_stiffness * (dt / substep)**2)

            for C_shape, C_init_shape in zip(softbody.C_shape_list, softbody.C_init_shape_list):
                self.L_list.append(torch.zeros_like(C_init_shape).to(cfg.device))
                self.project_list.append(project_C_shape_simple(
                    softbody.V_w, softbody.V_mass_no_inf, C_shape, C_init_shape, V_shape_compliance))
        
        if use_spring_boundary:
            self.V_boundary_stiffness = V_boundary_stiffness
            V_boundary_compliance = 1 / (V_boundary_stiffness * (dt / substep)**2)
            # for C_dist, C_init_d, C_lut_0, C_lut_1, C_V_0, C_V_1 in zip(softbody.C_boundary_list, softbody.C_init_boundary_d_list, softbody.C_boundary_lut_0, softbody.C_boundary_lut_1, softbody.C_boundary_V_0, softbody.C_boundary_V_1):
            #     # print(torch.zeros_like(C_init_d).shape)
                
            #     self.L_list.append(torch.zeros_like(C_init_d).to(cfg.device))
            #     # self.L_list.append(torch.zeros_like(C_lut[:, 1]).to(cfg.device))
            #     self.project_list.append(project_C_spring_boundary(
            #         softbody.V_w, V_boundary_compliance, C_dist, C_init_d, C_lut_0, C_lut_1, C_V_0, C_V_1
            #     ))

            for C_dist, C_init_d, C_mtx in zip(softbody.C_boundary_list, softbody.C_init_boundary_d_list, softbody.C_boundary_mtx):
                # print(torch.zeros_like(C_init_d).shape)
                # print(C_init_d)
                self.L_list.append(torch.zeros_like(C_init_d).to(cfg.device))
                # self.L_list.append(torch.zeros_like(C_lut[:, 1]).to(cfg.device))
                self.project_list.append(project_C_spring_boundary(
                    softbody.V_w, V_boundary_compliance, C_dist, C_init_d, C_mtx
                ))


        # grasp_constraints
        if hasattr(self.softbody, 'grasp_point'):
            for C_grasp, C_grasp_d in zip(softbody.C_grasp_list, softbody.C_grasp_d_list):

                self.L_list.append(torch.zeros_like(C_grasp_d).to(cfg.device))
                self.project_list.append(project_C_grasp(
                    softbody.V_w, C_grasp, C_grasp_d, softbody.grasp_point
                )) 

    # def deformation_registration(self, V)

    def forward_with_obs(self, V, V_velocity, obs):
        sub_dt = self.dt / self.substep
        for _ in range(self.substep):
            # update velocity
            V_velocity_predict = V_velocity + sub_dt * self.V_force / self.V_mass
            # update predict
            V_predict = V.clone()
            if not self.quasi_static:
                V_predict += sub_dt * V_velocity_predict
            else:
                V_predict += 0.5 * (self.V_force/self.V_mass) * (sub_dt**2)
            # set lagrange to 0
            self.L_list = [torch.zeros_like(L) for L in self.L_list]
            # solver iteration
            for _ in range(self.iteration):

                for i in range(len(self.L_list)):
                    V_predict, self.L_list[i] = \
                        self.project_list[i].forward(V_predict, self.L_list[i])
                
                # TODO: implement gradient descent on chamfer distance between V_predict and obs
                V_predict = utils.chamfer_distance_GD(V_predict, obs, self.softbody, sample_surface=True)

            # NOTE: ground contact code is outdated
            if hasattr(self.softbody, 'contact_field'):
                contact_field = self.softbody.contact_field
                V_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V[:, :-1])
                V_predict_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V_predict[:, :-1].detach())
            
                V_xy_contact = (
                    V_xy_norm * (contact_field.shape[0] - 1)).long()
                V_predict_xy_contact = (
                    V_predict_xy_norm * (contact_field.shape[1]-1)).long()
                V_xy_contact[:, 0] = torch.clamp(V_xy_contact[:, 0], 0, contact_field.shape[0]-1)
                V_xy_contact[:, 1] = torch.clamp(V_xy_contact[:, 1], 0, contact_field.shape[1]-1)
                V_predict_xy_contact[:, 0] = torch.clamp(V_predict_xy_contact[:, 0], 0, contact_field.shape[0]-1)
                V_predict_xy_contact[:, 1] = torch.clamp(V_predict_xy_contact[:, 1], 0, contact_field.shape[1]-1)
                V_contact_height = contact_field[V_xy_contact[:, 0], V_xy_contact[:, 1]]
                V_predict_contact_height = contact_field[V_predict_xy_contact[:, 0], V_predict_xy_contact[:, 1]]

                # collision detection
                col_idx = (V_predict[:, 2] < V_predict_contact_height) & (
                    V[:, 2] > V_contact_height)
                V_predict[col_idx, 2] = V_predict_contact_height[col_idx] + 1e-5

                vio_idx = (V_predict[:, 2] < V_predict_contact_height) & (
                    V[:, 2] < V_contact_height)
                V_predict[vio_idx, 2] = V_predict_contact_height[vio_idx] + 1e-5
            else:
                col_idx = (V_predict[:, 2] < self.plane_height) & (V[:, 2] > self.plane_height)
                # find the penetration point on ground plane
                h_prev = V[col_idx, 2] - self.plane_height
                h_after = self.plane_height - V_predict[col_idx, 2]
                V_predict[col_idx, 0] = V[col_idx, 0] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx, 0] - V[col_idx, 0])
                V_predict[col_idx, 1] = V[col_idx, 1] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx, 1] - V[col_idx, 1])
                V_predict[col_idx, 2] = self.plane_height + 1e-5
                # violation detection
                vio_idx = (V_predict[:, 2] < self.plane_height) & (V[:, 2] < self.plane_height)
                V_predict[vio_idx, 2] = self.plane_height + 1e-5
            
            # update actual V_velocity
            V_velocity = (V_predict - V) / sub_dt
            # modify velocity after collision
            V_velocity[col_idx, 2] = -V_velocity[col_idx, 2]
            # modify velocity after violation fix
            V_velocity[vio_idx] = torch.tensor([0.]).to(cfg.device)
            # update actual V
            V = V_predict.clone()

        return V, V_velocity

    def forward(self,
                V: torch.Tensor,
                V_velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Execute the prediction and constraint update step for all vertices and velocities

        Args:
            V (torch.Tensor): vertices in shape #vertices x 3
            V_velocity (torch.Tensor): velocities in shape #vertices x 3

        Returns:
            tuple(torch.Tensor, torch.Tensor): updated vertices and velocities
        '''

        sub_dt = self.dt / self.substep
        for _ in range(self.substep):
            # update velocity
            V_velocity_predict = V_velocity + sub_dt * self.V_force / self.V_mass
            # update predict
            V_predict = V.clone()
            if not self.quasi_static:
                V_predict += sub_dt * V_velocity_predict
            else:
                V_predict += 0.5 * (self.V_force/self.V_mass) * (sub_dt**2)
            # print(self.V_mass.min())
            # set lagrange to 0

            self.L_list = [torch.zeros_like(L).to(cfg.device) for L in self.L_list]

            # solver iteration
            # for _ in range(self.iteration):
            for iter in range(self.iteration):
                for i in range(len(self.L_list)):
                    V_predict, self.L_list[i] = \
                        self.project_list[i].forward(V_predict, self.L_list[i])
                    
                    if True in torch.isnan(V_predict): 
                        print("You hit a nan!!!")
                        print()
            # print(V==V_predict)

            if hasattr(self.softbody, 'contact_field'):
                contact_field = self.softbody.contact_field
                V_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V[:, :-1])
                V_predict_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V_predict[:, :-1].detach())          
                V_xy_contact = (
                    V_xy_norm * (contact_field.shape[0] - 1)).long()
                V_predict_xy_contact = (
                    V_predict_xy_norm * (contact_field.shape[1]-1)).long()
                V_xy_contact[:, 0] = torch.clamp(V_xy_contact[:, 0], 0, contact_field.shape[0]-1)
                V_xy_contact[:, 1] = torch.clamp(V_xy_contact[:, 1], 0, contact_field.shape[1]-1)
                V_predict_xy_contact[:, 0] = torch.clamp(V_predict_xy_contact[:, 0], 0, contact_field.shape[0]-1)
                V_predict_xy_contact[:, 1] = torch.clamp(V_predict_xy_contact[:, 1], 0, contact_field.shape[1]-1)
                V_contact_height = contact_field[V_xy_contact[:, 0], V_xy_contact[:, 1]]
                V_predict_contact_height = contact_field[V_predict_xy_contact[:, 0], V_predict_xy_contact[:, 1]]

                # collision detection
                col_idx = (V_predict[:, 2] < V_predict_contact_height) & (
                    V[:, 2] > V_contact_height)
                V_predict[col_idx, 2] = V_predict_contact_height[col_idx] + 1e-5

                vio_idx = (V_predict[:, 2] < V_predict_contact_height) & (
                    V[:, 2] < V_contact_height)
                V_predict[vio_idx, 2] = V_predict_contact_height[vio_idx] + 1e-5
            else:
                col_idx = (V_predict[:, 2] < self.plane_height) & (V[:, 2] > self.plane_height)
                # find the penetration point on ground plane
                h_prev = V[col_idx, 2] - self.plane_height
                h_after = self.plane_height - V_predict[col_idx, 2]
                V_predict[col_idx, 0] = V[col_idx, 0] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx, 0] - V[col_idx, 0])
                V_predict[col_idx, 1] = V[col_idx, 1] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx, 1] - V[col_idx, 1])
                V_predict[col_idx, 2] = self.plane_height + 1e-5
                # violation detection
                vio_idx = (V_predict[:, 2] < self.plane_height) & (V[:, 2] < self.plane_height)
                V_predict[vio_idx, 2] = self.plane_height + 1e-5

            # update actual V_velocity
            V_velocity = (V_predict - V) / sub_dt
            # modify velocity after collision
            V_velocity[col_idx, 2] = -V_velocity[col_idx, 2]
            # modify velocity after violation fix
            V_velocity[vio_idx] = torch.tensor([0.]).to(cfg.device)
            # update actual V
            V = V_predict.clone()

        return V, V_velocity


class project_C_grasp(torch.nn.Module):
    def __init__(self,
                 V_w: torch.Tensor,
                 C_grasp: torch.Tensor,
                 C_grasp_d: torch.Tensor,
                 grasp_point: torch.Tensor) -> None:
        super(project_C_grasp, self).__init__()
        self.V_w = V_w.detach().clone()
        # to optimize stiffness passed in, remove detach()
        self.C_grasp = C_grasp.detach().clone()
        self.C_grasp_d = C_grasp_d.detach().clone()
        self.grasp_point = grasp_point.detach().clone()
    
    def forward(self, V_predict, L):
        # position difference vectors
        # NOTE verify this indexing
        N = V_predict[self.C_grasp] - self.grasp_point
        # print(self.grasp_point)
        # distance
        D = torch.norm(N, p=2, dim=-1, keepdim=True)
        # constraint values
        C = D - self.C_grasp_d
        # normalized difference vectors
        N_norm = N / D
        # average compliance
        A = 1
        # weighted inverse mass

        # NOTE: make sure understand this S infinity
        S = self.V_w[self.C_grasp]+0
        S[S == 0] = torch.inf
        # delta lagrange
        L_delta = (-C - A * L) / (S + A)
        # new lagrange
        L_new = L + L_delta
        # new V_predict
        V_predict_new = V_predict.clone()
        # update for 0 vertex in constraint
        V_predict_new[self.C_grasp] += self.V_w[self.C_grasp] * L_delta * N_norm

        return V_predict_new, L_new

# project an independent set of distance constraints
class project_C_dist(torch.nn.Module):
    def __init__(self,
                 V_w: torch.Tensor,
                 V_compliance: torch.Tensor,
                 C_dist: torch.Tensor,
                 C_init_d: torch.Tensor) -> None:
        super(project_C_dist, self).__init__()
        self.V_w = V_w.detach().clone()
        # to optimize stiffness passed in, remove detach()add_thinshell
        self.V_compliance = V_compliance.detach().clone()
        self.C_dist = C_dist.detach().clone()
        self.C_init_d = C_init_d.detach().clone()
        

    def forward(self,
                V_predict: torch.Tensor,
                L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position difference vectors
        N = V_predict[self.C_dist[:, 0]] - V_predict[self.C_dist[:, 1]]
        # distance
        D = torch.norm(N, p=2, dim=1, keepdim=True)
        # constarint values
        C = D - self.C_init_d
        # normalized difference vectors
        N_norm = N / D
        # average compliance
        A = (self.V_compliance[self.C_dist[:, 0]] +
             self.V_compliance[self.C_dist[:, 1]]) / 2
        # weighted inverse mass
        S = self.V_w[self.C_dist[:, 0]] + self.V_w[self.C_dist[:, 1]]
        S[S == 0] = torch.inf
        # delta lagrange
        L_delta = (-C - A * L) / (S + A)
        
        # new lagrange
        L_new = L + L_delta
        # new V_predict
        V_predict_new = V_predict.clone()
        # update for 0 vertex in constraint
        # print(self.V_w[self.C_dist[:, 0]].shape, L_delta.shape, N_norm.shape, self.C_dist[:, 0].shape)
        V_predict_new[self.C_dist[:, 0]
                      ] += self.V_w[self.C_dist[:, 0]] * L_delta * N_norm
        # update for 1 vertex in constraint
        # print(True in torch.isnan(self.V_w[self.C_dist[:, 1]]), True in torch.isnan(L_delta), True in torch.isnan(N_norm))
        V_predict_new[self.C_dist[:, 1]
                      ] -= self.V_w[self.C_dist[:, 1]] * L_delta * N_norm

        return V_predict_new, L_new

## old
# class project_C_spring_boundary(torch.nn.Module):
#     def __init__(self,
#                  V_w: torch.Tensor,
#                  V_compliance: torch.Tensor,
#                  C_dist: torch.Tensor,
#                  C_init_d: torch.Tensor,
#                  C_lut: torch.Tensor) -> None:
#         super(project_C_spring_boundary, self).__init__()
#         self.V_w = V_w.detach().clone()
#         # to optimize stiffness passed in, remove detach()add_thinshell
#         self.V_compliance = V_compliance.detach().clone()
#         self.C_dist = C_dist.detach().clone()
#         self.C_init_d = C_init_d.detach().clone()

#     def forward(self,
#                 V_predict: torch.Tensor,
#                 L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # position difference vectors
#         N = V_predict[self.C_dist[:, 0]] - V_predict[self.C_dist[:, 1]]
#         # print('self.C_dist[:, 0]', self.C_dist[:, 0])
#         # print("self.C_dist[:, 1]", self.C_dist[:, 1])
#         # distance
#         D = torch.norm(N, p=2, dim=1, keepdim=True)
#         # constarint values
#         C = D - self.C_init_d
#         # normalized difference vectors
#         N_norm = N / (D+1e-6)
#         # average compliance
#         # A = self.V_compliance[self.C_dist[:, 0]]
#         # A = torch.ones_like(self.V_compliance).to(cfg.device)
#         A = self.V_compliance
            
#         # weighted inverse mass
#         S = self.V_w[self.C_dist[:, 0]] + self.V_w[self.C_dist[:, 1]]
#         S[S == 0] = torch.inf
#         # delta lagrange
#         L_delta = (-C - A * L) / (S + A)
#         # new lagrange
#         L_new = L + L_delta
#         # new V_predict
#         V_predict_new = V_predict.clone()
#         # print(self.C_init_d.shape)
#         # update for 0 vertex in constraint
#         V_predict_new[self.C_dist[:, 0]
#                       ] += self.V_w[self.C_dist[:, 0]] * L_delta * N_norm
#         # update for 1 vertex in constraint
#         V_predict_new[self.C_dist[:, 1]
#                       ] -= self.V_w[self.C_dist[:, 1]] * L_delta * N_norm

#         return V_predict_new, L_new

class project_C_spring_boundary(torch.nn.Module):
    def __init__(self,
                 V_w: torch.Tensor,
                 V_compliance: torch.Tensor,
                 C_dist: torch.Tensor,
                 C_init_d: torch.Tensor,
                 C_mtx: torch.tensor) -> None:
                #  C_lut_0: list,
                #  C_lut_1: list,
                #  C_V_0: torch.tensor,
                #  C_V_1: torch.tensor) -> None:
        super(project_C_spring_boundary, self).__init__()
        self.V_w = V_w.detach().clone()
        # to optimize stiffness passed in, remove detach()add_thinshell
        # self.V_compliance = V_compliance.detach().clone()
        self.V_compliance = V_compliance.clone()
        self.C_dist = C_dist.detach().clone()
        self.C_init_d = C_init_d.detach().clone()
        self.C_mtx = C_mtx.detach().clone()
        # self.C_lut_0 = C_lut_0.copy()
        # self.C_lut_1 = C_lut_1.copy()
        # self.C_V_0 = C_V_0.detach().clone()
        # self.C_V_1 = C_V_1.detach().clone()


    def forward(self,
                V_predict: torch.Tensor,
                L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position difference vectors
        N = V_predict[self.C_dist[:, 0]] - V_predict[self.C_dist[:, 1]]
        # print('self.C_dist[:, 0]', self.C_dist[:, 0])
        # print("self.C_dist[:, 1]", self.C_dist[:, 1])
        # distance
        D = torch.norm(N, p=2, dim=1, keepdim=True)
        # constarint values
        C = D - self.C_init_d
        # print(C.max())
        # print(N.shape)
        # normalized difference vectors
        N_norm = N / D
        # average compliance
        # A = self.V_compliance[self.C_dist[:, 0]]
        # A = torch.ones_like(self.V_compliance).to(cfg.device)
        A = self.V_compliance
            
        # weighted inverse mass
        S = self.V_w[self.C_dist[:, 0]] + self.V_w[self.C_dist[:, 1]]
        S[S == 0] = torch.inf
        # delta lagrange
        L_delta = (-C - A * L) / (S + A)
        # print(torch.sum(L_delta > 1e-10))
        # new lagrange
        L_new = L + L_delta
        # new V_predict
        V_predict_new = V_predict.clone()
        # print(self.C_init_d.shape)
        # update for 0 vertex in constraint
        # L_0 = torch.zeros((self.C_V_0.shape[0], 3)).to(cfg.device)
        # for i in range(len(self.C_lut_0)):
        #     L_0[i] = torch.mean(L_delta[self.C_lut_0[i]] * N_norm[self.C_lut_0[i]], 0)

        # L_1 = torch.zeros((self.C_V_1.shape[0], 3)).to(cfg.device)
        # for i in range(len(self.C_lut_1)):
        #     L_1[i] = torch.mean(L_delta[self.C_lut_1[i]] * N_norm[self.C_lut_1[i]], 0)
        L_temp = torch.transpose(L_delta * N_norm, 0, 1).type(torch.FloatTensor).to(cfg.device)
        # print(L_temp.shape)
        L = torch.zeros((self.C_mtx.shape[1], 3)).to(cfg.device)
        L[:, 0] = L_temp[0] @ self.C_mtx
        L[:, 1] = L_temp[1] @ self.C_mtx
        L[:, 2] = L_temp[2] @ self.C_mtx
        # print(torch.sum(L[self.C_dist[:, 0]] > 1e-6))
        # L = torch.transpose(L, 0, 1)
        # print(L.shape)

        # print(self.V_w[self.C_V_0].shape, L_0.shape)
        
        # V_predict_new[self.C_V_0
        #               ] += self.V_w[self.C_V_0] * L_0
        # # update for 1 vertex in constraint
        # V_predict_new[self.C_V_1
        #               ] -= self.V_w[self.C_V_1] * L_1
        V_predict_new += self.V_w * L
        # print(L[self.C_dist[:, 0]])
        return V_predict_new, L_new


# class project_C_shape_simple(torch.nn.Module):

#     def __init__(self, V_w: torch.Tensor, V_mass_no_inf: torch.Tensor, C_shape: torch.Tensor, C_init_shape: torch.Tensor, V_compliance: float) -> None:
#         super(project_C_shape_simple, self).__init__()
#         self.V_w = V_w.detach().clone()
#         self.V_mass_no_inf = V_mass_no_inf.detach().clone()
#         self.C_shape = C_shape.detach().clone()
#         self.C_init_shape = C_init_shape.detach().clone()
#         self.V_compliance = V_compliance.detach().clone()
#         # self.NUM_C = self.C_shape.shape[0]
#         # self.NUM_particles = self.C_shape.shape[1]

#         # self.svd_diff = torch_utils.svdv2.apply
#         # self.svd_diff = torch_utils.SVD_decomposition.apply

#     def forward(self, V_predict: torch.Tensor, L_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         v_pred = V_predict[self.C_shape].double()
#         v_mass = self.V_mass_no_inf[self.C_shape].double()
#         local_vec_init = self.C_init_shape.double()

#         self.v_mass = v_mass
#         self.local_vec_init = local_vec_init

#         C_value = self.compute_constraints_delta_batch_value(v_pred)
#         # # a simple way : to directly update delta_pos
#         stiffness = 1.0 / self.V_compliance[self.C_shape].double()  # B*N*1

#         inv_mass = self.V_w[self.C_shape].double()  # B*N*1
#         # print(C_value)
#         delta_pos = inv_mass.repeat(1, 1, 3) * C_value * stiffness.repeat(1, 1, 3)
#         # print(C_value)
#         # new V_predict
#         V_predict_new = V_predict.clone()
#         V_predict_new[self.C_shape] += delta_pos.squeeze()

#         L_new = L_last

#         # pdb.set_trace()

#         return V_predict_new, L_new

#     def compute_constraints_delta_batch_value(self, x_pos):

#         local_vec_init = self.local_vec_init
#         x_mass = self.v_mass
#         # print(local_vec_init, x_pos)
#         # compute COM of current prediction steps
#         weighted_pos = torch.mul(x_mass, x_pos)
#         curr_com = torch.sum(weighted_pos, dim=0) / torch.sum(x_mass, dim=0)
#         # local vectors from COM to each points
#         local_vec_pred = x_pos - curr_com
#         # print('pred', local_vec_pred)
#         # print('init', local_vec_init)
#         # wght_mtx = torch.diag_embed(x_mass.squeeze(2))
#         wght_mtx = torch.diag_embed(x_mass.squeeze())
#         # S_mtx = torch.matmul(torch.transpose(local_vec_init, 1, 2), torch.matmul(wght_mtx, local_vec_pred))
#         S_mtx = (local_vec_init.transpose(0, 1)) @ (wght_mtx @ local_vec_pred)
#         # check svd details : https://zhuanlan.zhihu.com/p/459933370
#         U, S, V_h = torch.linalg.svd(S_mtx)
#         # U, S, U_h = torch.linalg.svd(S_mtx)
#         # U_h, S = self.svd_diff(S_mtx)
#         # print(S)
#         U_h = torch.transpose(U, 0, 1)
#         V = torch.transpose(V_h, 0, 1)
#         # pdb.set_trace()
#         # det = torch.det(torch.matmul(U, U_h))
#         det = torch.linalg.det(torch.matmul(V, U_h))
#         # det = det.view(-1, 1, 1)
#         # print(U_h[:, :2], U_h[:, -1] * det)
#         V_det = torch.cat((V[:, :2], V[:, -1:] * det), 1)
#         # print(U_T)
#         rot_mtx = torch.matmul(V_det, U_h)
#         # print(rot_mtx)
#         delta_x = (rot_mtx @ local_vec_init.transpose(0, 1)).transpose(0, 1) + (-local_vec_pred)
#         # print(rot_mtx)
#         return delta_x

class project_C_shape_simple(torch.nn.Module):

    def __init__(self, V_w: torch.Tensor, V_mass_no_inf: torch.Tensor, C_shape: torch.Tensor, C_init_shape: torch.Tensor, V_compliance: float) -> None:
        super(project_C_shape_simple, self).__init__()
        self.V_w = V_w.detach().clone()
        self.V_mass_no_inf = V_mass_no_inf.detach().clone()
        self.C_shape = C_shape.detach().clone()
        self.C_init_shape = C_init_shape.detach().clone()
        self.V_compliance = V_compliance.detach().clone()
        self.NUM_C = self.C_shape.shape[0]
        self.NUM_particles = self.C_shape.shape[1]

        # self.svd_diff = torch_utils.svdv2.apply
        # self.svd_diff = torch_utils.SVD_decomposition.apply

    def forward(self, V_predict: torch.Tensor, L_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_pred = V_predict[self.C_shape].double()
        v_mass = self.V_mass_no_inf[self.C_shape].double()
        local_vec_init = self.C_init_shape.double()
        # print(self.C_init_shape)
        self.v_mass = v_mass
        self.local_vec_init = local_vec_init

        C_value = self.compute_constraints_delta_batch_value(v_pred)

        # # a simple way : to directly update delta_pos
        stiffness = 1.0 / self.V_compliance[self.C_shape].double()  # B*N*1


        inv_mass = self.V_w[self.C_shape].double()  # B*N*1
        delta_pos = inv_mass.repeat(1, 1, 3) * C_value * stiffness.repeat(1, 1, 3)


        # new V_predict
        V_predict_new = V_predict.clone()
        V_predict_new[self.C_shape] += delta_pos

        L_new = L_last

        # pdb.set_trace()

        return V_predict_new, L_new

    def compute_constraints_delta_batch_value(self, x_pos):

        local_vec_init = self.local_vec_init
        x_mass = self.v_mass

        # compute COM of current prediction steps
        weighted_pos = torch.mul(x_mass, x_pos)
        curr_com = torch.sum(weighted_pos, dim=1) / torch.sum(x_mass, dim=1)
        # local vectors from COM to each points
        local_vec_pred = x_pos - curr_com.unsqueeze(1)
        wght_mtx = torch.diag_embed(x_mass.squeeze(2))
        # print('pred', local_vec_pred)
        # print('init', local_vec_init)
        # S_mtx = torch.matmul(torch.transpose(local_vec_init, 1, 2), torch.matmul(wght_mtx, local_vec_pred))
        S_mtx = (local_vec_init.transpose(1, 2)) @ (wght_mtx @ local_vec_pred)
        # check svd details : https://zhuanlan.zhihu.com/p/459933370
        _, S, U_h = torch.linalg.svd(S_mtx)
        # U, S, U_h = torch.linalg.svd(S_mtx)
        # U_h, S = self.svd_diff(S_mtx)
        # print(S)
        U = torch.transpose(U_h, 1, 2)
        # pdb.set_trace()
        # det = torch.det(torch.matmul(U, U_h))
        det = torch.linalg.det(torch.matmul(U, U_h))
        det = det.view(-1, 1, 1)
        U_T = torch.cat((U_h[:, :2, :], U_h[:, -1:, :] * det), 1)

        rot_mtx = torch.matmul(U, U_T)
        delta_x = (rot_mtx @ local_vec_init.transpose(1, 2)).transpose(1, 2) + (-local_vec_pred)
        return delta_x

def get_energy_thinshell(softbody: XPBDSoftbody,
                         V_predict: torch.Tensor,
                         V_dist_stiffness: torch.Tensor,
                         mask: set = None) -> torch.Tensor:
    dist_C, dist_C_stiffness = __get_dist_constraints(softbody,
                                                      V_predict,
                                                      V_dist_stiffness,
                                                      mask)
    # energy is C^2 * stiffness / 2
    dist_energy = torch.square(dist_C) * dist_C_stiffness / 2
    return torch.mean(dist_energy)

def __get_dist_constraints(softbody: XPBDSoftbody,
                           V_predict: torch.Tensor,
                           V_stiffness: torch.Tensor,
                           mask: set = None) -> torch.Tensor:
    C = []
    C_stiffness = []
    # collect all distance constraints
    for C_dist, C_init_d in zip(softbody.C_dist_list, softbody.C_init_d_list):
        if mask == None or (C_dist[:, 0] in mask and C_dist[:, 1] in mask):
            # position difference vectors
            N = V_predict[C_dist[:, 0]] - V_predict[C_dist[:, 1]]
            # distance
            D = torch.norm(N, p=2, dim=1, keepdim=True)
            # constarint values
            C.append(D - C_init_d)
            # average stiffness
            C_stiffness.append((V_stiffness[C_dist[:, 0]] +
                                V_stiffness[C_dist[:, 1]]) / 2)
    return torch.cat(C), torch.cat(C_stiffness)

def get_energy_boundary(softbody: XPBDSoftbody,
                         V_predict: torch.Tensor,
                         V_boundary_stiffness: torch.Tensor,
                         mask: set = None) -> torch.Tensor:
    
    V_boundary_stiffness_threshold = V_boundary_stiffness.clone()
    # V_boundary_stiffness_threshold[V_boundary_stiffness_threshold < 1e-3] = 0
    V_boundary_stiffness_threshold = V_boundary_stiffness_threshold * torch.sigmoid(1e5 * (V_boundary_stiffness_threshold - 1e-3))
    # dist_C, dist_C_stiffness = __get_spring_boundary_constraints(softbody,
    #                                                   V_predict,
    #                                                   V_boundary_stiffness_threshold,
    #                                                   mask)
    # Since this version for the boundary constrain, it is stiffness - constrain not stiffness - vertex
    # The stiffness can simply grap from the input
    dist_C = __get_spring_boundary_constraints(softbody,
                                                      V_predict,
                                                      V_boundary_stiffness_threshold,
                                                      mask)
    # energy is C^2 * stiffness / 2
    # dist_C_stiffness = V_boundary_stiffness_threshold.clone()
    boundary_energy = torch.square(dist_C) * V_boundary_stiffness_threshold / 2
    return boundary_energy

def __get_spring_boundary_constraints(softbody, V_predict, V_boundary_stiffness, mask=None):
    C = []
    # C_stiffness = []
    # collect all distance constraints
    for C_dist, C_init_d in zip(softbody.C_boundary_list, softbody.C_init_boundary_d_list):
        if mask == None or (C_dist[:, 0] in mask and C_dist[:, 1] in mask):
            # position difference vectors
            N = V_predict[C_dist[:, 0]] - V_predict[C_dist[:, 1]]
            # distance
            D = torch.norm(N, p=2, dim=1, keepdim=True)
            # constarint values
            C.append(D - C_init_d)
            # average stiffness
            # C_stiffness.append(V_boundary_stiffness[C_dist[:, 0]])
    return torch.cat(C)#, torch.cat(C_stiffness)