import torch
from typing import Tuple
from xpbd_softbody import XPBDSoftbody
import config as cfg
import utils



class XPBDStepBatch(torch.nn.Module):
    def __init__(self, softbody: XPBDSoftbody,
                 V_dist_stiffness: torch.Tensor,
                 V_boundary_stiffness: torch.Tensor,
                 dt: float = 0.01, substep: int = 1, iteration: int = 10,
                 plane_height=torch.tensor([-torch.inf]).to(cfg.device),
                 quasi_static=False,
                 multi_grasp=False,
                 C_grasp=None,
                 C_grasp_d=None,
                 boundary_list=None,
                 use_boundary=False) -> None:
        '''
        Initialize step function

        Args:
            softbody (XPBDSotbody): softbody object
            dist_stiffness (torch.Tensor): distance stiffness
            vol_stiffness (torch.Tensor): volume stiffness
            dt (float): step size
            substep: substep count
            iteration (int): solver iteration count
            td_stiffness (float): tendon stiffness
        '''
        super(XPBDStepBatch, self).__init__()
        self.B = V_dist_stiffness.shape[0]
        # state tensors
        self.V_mass = softbody.V_mass.expand(self.B, -1, -1).clone()
        self.V_force = softbody.V_force.clone().expand(self.B, -1, -1)
        self.V_w = softbody.V_w.expand(self.B, -1, -1).clone()

        # bc different batch can have different boundary condition, so this V_w and V_mass need to be modified
        if multi_grasp:
            assert(len(boundary_list) == self.B)
            for i in range(self.B):    
                self.V_mass[i, boundary_list[i]] = torch.inf
                self.V_w[i, boundary_list[i]] = 0

        # solver parameters
        self.dt = dt
        self.substep = substep
        self.iteration = iteration
        self.plane_height = plane_height
        # constraint projection layers
        self.project_list = []
        self.L_list = []
        self.quasi_static = quasi_static
        self.softbody = softbody
        # distance constraints
        V_dist_compliance = 1 / (V_dist_stiffness * (dt / substep)**2)
        for C_dist, C_init_d in zip(softbody.C_dist_list, softbody.C_init_d_list):
            self.L_list.append(torch.zeros(self.B, *C_init_d.shape).to(cfg.device))
            self.project_list.append(project_C_dist_batch(
                 self.V_w, V_dist_compliance, C_dist, C_init_d
            ))
        
        # spring boundary constraints
        V_boundary_compliance = 1 / (V_boundary_stiffness * (dt / substep)**2)
        # for C_dist, C_init_d in zip(softbody.C_boundary_list, softbody.C_init_boundary_d_list):
        #     self.L_list.append(torch.zeros(self.B, *C_init_d.shape).to(cfg.device))
        #     self.project_list.append(project_C_spring_boundary_batch(
        #          self.V_w, V_boundary_compliance, C_dist, C_init_d
        #     ))
        if use_boundary:
            for C_dist, C_init_d, C_mtx in zip(softbody.C_boundary_list, softbody.C_init_boundary_d_list, softbody.C_boundary_mtx):
                # print(torch.zeros_like(C_init_d).shape)
                # print(C_init_d)
                # print(*C_init_d.shape)
                self.L_list.append(torch.zeros(self.B, *C_init_d.shape).to(cfg.device))
                # self.L_list.append(torch.zeros_like(C_lut[:, 1]).to(cfg.device))
                self.project_list.append(project_C_spring_boundary_batch(
                    self.V_w, V_boundary_compliance, C_dist, C_init_d, C_mtx
                ))

        # if not multi_grasp:
        #     if hasattr(softbody, 'grasp_point'):
        #         for C_grasp, C_grasp_d in zip(softbody.C_grasp_list, softbody.C_grasp_d_list):
        #             self.L_list.append(torch.zeros_like(C_grasp_d).to(cfg.device))
        #             self.project_list.append(project_C_grasp_batch(
        #                 self.V_w, C_grasp, C_grasp_d
        #             )) 
        # else:
        #     self.L_list.append(torch.zeros_like(C_grasp_d).to(cfg.device))
        #     self.project_list.append(project_C_grasp_batch_multi(
        #                 self.V_w, C_grasp, C_grasp_d
        #             )) 

    def forward_parallel(self,
                V: torch.Tensor,
                V_velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert V.shape[0] == self.B
        assert V_velocity.shape[0] == self.B
        # assert grasp_points.shape[0] == self.B

        sub_dt = self.dt / self.substep
        for _ in range(self.substep):
            # update predict
            V_predict = V.clone()
            if not self.quasi_static:
                # update velocity
                V_velocity_predict = V_velocity + sub_dt * self.V_force / self.V_mass
                V_predict += sub_dt * V_velocity_predict
            else:
                V_predict += 0.5 * (self.V_force/self.V_mass) * (sub_dt**2)

            # set lagrange to 0
            self.L_list = [torch.zeros_like(L).to(cfg.device) for L in self.L_list]
            # solver iteration
            for _ in range(self.iteration):
                for i in range(len(self.L_list)):
                    V_predict, self.L_list[i] = \
                        self.project_list[i].forward(V_predict, self.L_list[i])
            

            if hasattr(self.softbody, 'contact_field'):
                contact_field = self.softbody.contact_field
                V_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V[..., :-1])
                V_predict_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V_predict[..., :-1].detach())
                
                V_xy_contact = (
                    V_xy_norm * (contact_field.shape[0] - 1)).long()
                V_predict_xy_contact = (
                    V_predict_xy_norm * (contact_field.shape[1]-1)).long()

                V_xy_contact[..., 0] = torch.clamp(V_xy_contact[..., 0], 0, contact_field.shape[0]-1)
                V_xy_contact[..., 1] = torch.clamp(V_xy_contact[..., 1], 0, contact_field.shape[1]-1)
                V_predict_xy_contact[..., 0] = torch.clamp(V_predict_xy_contact[..., 0], 0, contact_field.shape[0]-1)
                V_predict_xy_contact[..., 1] = torch.clamp(V_predict_xy_contact[..., 1], 0, contact_field.shape[1]-1)
                V_contact_height = contact_field[V_xy_contact[..., 0], V_xy_contact[..., 1]]
                V_predict_contact_height = contact_field[V_predict_xy_contact[..., 0], V_predict_xy_contact[..., 1]]

                col_idx = (V_predict[..., 2] < V_predict_contact_height) & (
                    V[..., 2] > V_contact_height)
                col_idx_ = col_idx.nonzero()
                V_predict[col_idx_[:, 0], col_idx_[:, 1], 2] = V_predict_contact_height[col_idx_[:, 0], col_idx_[:, 1]] + 1e-5

                vio_idx = (V_predict[..., 2] < V_predict_contact_height) & (
                    V[..., 2] < V_contact_height)
                vio_idx_ = vio_idx.nonzero()
                V_predict[vio_idx_[:, 0], vio_idx_[:, 1], 2] = V_predict_contact_height[vio_idx_[:, 0], vio_idx_[:, 1]] + 1e-5

            else:
                col_idx = (V_predict[..., 2] < self.plane_height) & (V[..., 2] > self.plane_height)
                col_idx_ = col_idx.nonzero()

                h_prev = V[col_idx_[:, 0], col_idx_[:, 1], 2] - self.plane_height
                h_after = self.plane_height - V_predict[col_idx_[:, 0], col_idx_[:, 1], 2]

                V_predict[col_idx_[:, 0], col_idx_[:, 1], 0] = V[col_idx_[:, 0], col_idx_[:, 1], 0] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx_[:, 0], col_idx_[:, 1], 0]
                                                        - V[col_idx_[:, 0], col_idx_[:, 1], 0])
                
                V_predict[col_idx_[:, 0], col_idx_[:, 1], 1] = V[col_idx_[:, 0], col_idx_[:, 1], 1] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx_[:, 0], col_idx_[:, 1], 1] - V[col_idx_[:, 0], col_idx_[:, 1], 1])
                V_predict[col_idx_[:, 0], col_idx_[:, 1], 2] = self.plane_height + 1e-5

                vio_idx = (V_predict[..., 2] < self.plane_height) & (V[..., 2] < self.plane_height)
                vio_idx_ = vio_idx.nonzero()

                V_predict[vio_idx_[:, 0], vio_idx_[:, 1], 2] = self.plane_height + 1e-5

            # h_prev = V[col_idx] - self.plane_height

            # update actual V_velocity
            V_velocity = (V_predict - V) / sub_dt

            V_velocity[col_idx_[:, 0], col_idx_[:, 1], 2] = -V_velocity[col_idx_[:, 0], col_idx_[:, 1], 2]
            V_velocity[vio_idx_[:, 0], vio_idx_[:, 1], 2] = torch.tensor([0.]).to(cfg.device)

            V = V_predict.clone()

        return V, V_velocity
    
    def forward_parallel_with_obs(self,
                V: torch.Tensor,
                V_velocity: torch.Tensor,
                grasp_points: torch.Tensor,
                obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert V.shape[0] == self.B
        assert V_velocity.shape[0] == self.B
        assert grasp_points.shape[0] == self.B

        sub_dt = self.dt / self.substep
        for _ in range(self.substep):
            # update predict
            V_predict = V.clone()
            if not self.quasi_static:
                # update velocity
                V_velocity_predict = V_velocity + sub_dt * self.V_force / self.V_mass
                V_predict += sub_dt * V_velocity_predict
            else:
                V_predict += 0.5 * (self.V_force/self.V_mass) * (sub_dt**2)

            # set lagrange to 0
            self.L_list = [torch.zeros_like(L).to(cfg.device) for L in self.L_list]
            # solver iteration
            for _ in range(self.iteration):
                for i in range(len(self.L_list)):
                    V_predict, self.L_list[i] = \
                        self.project_list[i].forward(V_predict, self.L_list[i], grasp_points)
                    
                    V_predict = utils.chamfer_distance_GD(V_predict, obs, self.softbody, sample_surface=True)
            

            if hasattr(self.softbody, 'contact_field'):
                contact_field = self.softbody.contact_field
                V_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V[..., :-1])
                V_predict_xy_norm = self.softbody.convert_to_normalize_coordinate(
                    V_predict[..., :-1].detach())
                
                V_xy_contact = (
                    V_xy_norm * (contact_field.shape[0] - 1)).long()
                V_predict_xy_contact = (
                    V_predict_xy_norm * (contact_field.shape[1]-1)).long()

                V_xy_contact[..., 0] = torch.clamp(V_xy_contact[..., 0], 0, contact_field.shape[0]-1)
                V_xy_contact[..., 1] = torch.clamp(V_xy_contact[..., 1], 0, contact_field.shape[1]-1)
                V_predict_xy_contact[..., 0] = torch.clamp(V_predict_xy_contact[..., 0], 0, contact_field.shape[0]-1)
                V_predict_xy_contact[..., 1] = torch.clamp(V_predict_xy_contact[..., 1], 0, contact_field.shape[1]-1)
                V_contact_height = contact_field[V_xy_contact[..., 0], V_xy_contact[..., 1]]
                V_predict_contact_height = contact_field[V_predict_xy_contact[..., 0], V_predict_xy_contact[..., 1]]

                col_idx = (V_predict[..., 2] < V_predict_contact_height) & (
                    V[..., 2] > V_contact_height)
                col_idx_ = col_idx.nonzero()
                V_predict[col_idx_[:, 0], col_idx_[:, 1], 2] = V_predict_contact_height[col_idx_[:, 0], col_idx_[:, 1]] + 1e-5

                vio_idx = (V_predict[..., 2] < V_predict_contact_height) & (
                    V[..., 2] < V_contact_height)
                vio_idx_ = vio_idx.nonzero()
                V_predict[vio_idx_[:, 0], vio_idx_[:, 1], 2] = V_predict_contact_height[vio_idx_[:, 0], vio_idx_[:, 1]] + 1e-5

            else:
                col_idx = (V_predict[..., 2] < self.plane_height) & (V[..., 2] > self.plane_height)
                col_idx_ = col_idx.nonzero()

                h_prev = V[col_idx_[:, 0], col_idx_[:, 1], 2] - self.plane_height
                h_after = self.plane_height - V_predict[col_idx_[:, 0], col_idx_[:, 1], 2]

                V_predict[col_idx_[:, 0], col_idx_[:, 1], 0] = V[col_idx_[:, 0], col_idx_[:, 1], 0] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx_[:, 0], col_idx_[:, 1], 0]
                                                        - V[col_idx_[:, 0], col_idx_[:, 1], 0])
                
                V_predict[col_idx_[:, 0], col_idx_[:, 1], 1] = V[col_idx_[:, 0], col_idx_[:, 1], 1] + \
                    (h_prev / (h_prev + h_after)) * (V_predict[col_idx_[:, 0], col_idx_[:, 1], 1] - V[col_idx_[:, 0], col_idx_[:, 1], 1])
                V_predict[col_idx_[:, 0], col_idx_[:, 1], 2] = self.plane_height + 1e-5

                vio_idx = (V_predict[..., 2] < self.plane_height) & (V[..., 2] < self.plane_height)
                vio_idx_ = vio_idx.nonzero()

                V_predict[vio_idx_[:, 0], vio_idx_[:, 1], 2] = self.plane_height + 1e-5

            # h_prev = V[col_idx] - self.plane_height

            # update actual V_velocity
            V_velocity = (V_predict - V) / sub_dt

            V_velocity[col_idx_[:, 0], col_idx_[:, 1], 2] = -V_velocity[col_idx_[:, 0], col_idx_[:, 1], 2]
            V_velocity[vio_idx_[:, 0], vio_idx_[:, 1], 2] = torch.tensor([0.]).to(cfg.device)

            V = V_predict.clone()

        return V, V_velocity


# project an independent set of distance constraints
class project_C_dist_batch(torch.nn.Module):
    def __init__(self,
                 V_w: torch.Tensor,
                 V_compliance: torch.Tensor,
                 C_dist: torch.Tensor,
                 C_init_d: torch.Tensor) -> None:
        super(project_C_dist_batch, self).__init__()
        self.V_w = V_w
        # to optimize stiffness passed in, remove detach()
        self.V_compliance = V_compliance.detach().clone()
        self.C_dist = C_dist.detach().clone()
        self.C_init_d = C_init_d.detach().clone()

    def forward(self,
                V_predict: torch.Tensor,
                L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position difference vectors
        N = V_predict[:, self.C_dist[:, 0]] - V_predict[:, self.C_dist[:, 1]]
        # distance
        D = torch.norm(N, p=2, dim=-1, keepdim=True)
        # constarint values
        C = D - self.C_init_d
        # normalized difference vectors
        N_norm = N / D

        A = (self.V_compliance[:, self.C_dist[:, 0]] +
            self.V_compliance[:, self.C_dist[:, 1]]) / 2

        
        # weighted inverse mass
        S = self.V_w[:, self.C_dist[:, 0]] + self.V_w[:, self.C_dist[:, 1]]
        S[S == 0] = torch.inf
        # delta lagrange
        L_delta = (-C - A * L) / (S + A)
        # new lagrange
        L_new = L + L_delta
        # new V_predict
        V_predict_new = V_predict.clone()
        # update for 0 vertex in constraint
        V_predict_new[:, self.C_dist[:, 0]] += self.V_w[:, self.C_dist[:, 0]] * L_delta * N_norm
        # update for 1 vertex in constraint
        V_predict_new[:, self.C_dist[:, 1]] -= self.V_w[:, self.C_dist[:, 1]] * L_delta * N_norm

        return V_predict_new, L_new


class project_C_spring_boundary_batch(torch.nn.Module):
    def __init__(self,
                 V_w: torch.Tensor,
                 V_compliance: torch.Tensor,
                 C_dist: torch.Tensor,
                 C_init_d: torch.Tensor,
                 C_mtx: torch.tensor) -> None:
        super(project_C_spring_boundary_batch, self).__init__()
        self.V_w = V_w.detach().clone()
        # to optimize stiffness passed in, remove detach()add_thinshell
        self.V_compliance = V_compliance.detach().clone()
        self.C_dist = C_dist.detach().clone()
        self.C_init_d = C_init_d.detach().clone()
        self.C_mtx = C_mtx.detach().clone()

    def forward(self,
                V_predict: torch.Tensor,
                L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position difference vectors
        N = V_predict[:, self.C_dist[:, 0]] - V_predict[:, self.C_dist[:, 1]]
        # distance
        D = torch.norm(N, p=2, dim=-1, keepdim=True)
        # constarint values
        C = D - self.C_init_d
        # normalized difference vectors
        N_norm = N / (D+1e-8)
        # average compliance
        # A = self.V_compliance[:, self.C_dist[:, 0]]
        A = self.V_compliance
            
        # weighted inverse mass
        S = self.V_w[:, self.C_dist[:, 0]] + self.V_w[:, self.C_dist[:, 1]]
        S[S == 0] = torch.inf
        # delta lagrange
        L_delta = (-C - A * L) / (S + A)
        # new lagrange
        L_new = L + L_delta
        # compute avg L
        # print(A.shape, L_delta.shape, N_norm.shape)
        L_temp = torch.transpose(L_delta * N_norm, 1, 2).type(torch.FloatTensor).to(cfg.device)
        # print(L_temp.shape)
        L = torch.zeros((self.V_compliance.shape[0], self.C_mtx.shape[1], 3)).to(cfg.device)
        L[:, :, 0] = L_temp[:, 0] @ self.C_mtx
        L[:, :, 1] = L_temp[:, 1] @ self.C_mtx
        L[:, :, 2] = L_temp[:, 2] @ self.C_mtx
        # new V_predict
        V_predict_new = V_predict.clone()
        # # update for 0 vertex in constraint
        # V_predict_new[:, self.C_dist[:, 0]] += self.V_w[:, self.C_dist[:, 0]] * L_delta * N_norm
        # # update for 1 vertex in constraint
        # V_predict_new[:, self.C_dist[:, 1]] -= self.V_w[:, self.C_dist[:, 1]] * L_delta * N_norm
        V_predict_new += self.V_w * L

        return V_predict_new, L_new

class project_C_grasp_batch(torch.nn.Module):
    def __init__(self,
                 V_w: torch.Tensor,
                 C_grasp: torch.Tensor,
                 C_grasp_d: torch.Tensor) -> None:
        super(project_C_grasp_batch, self).__init__()
        self.V_w = V_w.detach().clone()
        self.C_grasp = C_grasp.detach().clone()
        self.C_grasp_d = C_grasp_d.detach().clone()
    
    def forward(self, V_predict, L, grasp_points):
        # position difference vectors
        N = V_predict[:, self.C_grasp] - grasp_points
        # distance
        D = torch.norm(N, p=2, dim=-1, keepdim=True)
        # constarint values
        C = D - self.C_grasp_d
        # normalized difference vectors
        N_norm = N / D
        A = 1e2

        # weighted inverse mass
        S = self.V_w[:, self.C_grasp]+0
        S[S == 0] = torch.inf
        # delta lagrange
        L_delta = (-C - A * L) / (S + A)
        # new lagrange
        L_new = L + L_delta
        # new V_predict
        V_predict_new = V_predict.clone()
        # update for 0 vertex in constraint
        V_predict_new[:, self.C_grasp] += self.V_w[:, self.C_grasp] * L_delta * N_norm


        return V_predict_new, L_new
    
class project_C_grasp_batch_multi(torch.nn.Module):
    def __init__(self,
                 V_w: torch.Tensor,
                 C_grasp: torch.Tensor,
                 C_grasp_d: torch.Tensor) -> None:
        super(project_C_grasp_batch_multi, self).__init__()
        self.V_w = V_w.detach().clone()

        # C_grasp: B x N
        # C_grasp_d: B x N x 1
        self.C_grasp = C_grasp.detach().clone()
        self.C_grasp_d = C_grasp_d.detach().clone()


    def forward(self, V_predict, L, grasp_points):
        # position difference vectors
        B = V_predict.shape[0]
        grasp_connections = self.C_grasp.shape[-1]
        B_index = torch.arange(B).repeat_interleave(grasp_connections).reshape(self.C_grasp.shape).to(cfg.device)
        N = V_predict[B_index, self.C_grasp] - grasp_points
        # distance
        D = torch.norm(N, p=2, dim=-1, keepdim=True)
        # constarint values
        C = D - self.C_grasp_d
        # normalized difference vectors
        N_norm = N / D
        A = 1e2

        # weighted inverse mass
        S = self.V_w[B_index, self.C_grasp]+0
        S[S == 0] = torch.inf
        # delta lagrange
        L_delta = (-C - A * L) / (S + A)
        # new lagrange
        L_new = L + L_delta
        # new V_predict
        V_predict_new = V_predict.clone()
        # update for 0 vertex in constraint
        V_predict_new[B_index, self.C_grasp] += self.V_w[B_index, self.C_grasp] * L_delta * N_norm


        return V_predict_new, L_new
    

# NOTE: V_predict is batched, V_boundary_stiffness is not
def get_energy_boundary_batch(softbody: XPBDSoftbody,
                         V_predict: torch.Tensor,
                         V_boundary_stiffness: torch.Tensor,
                         mask: set = None) -> torch.Tensor:
    V_boundary_stiffness_threshold = V_boundary_stiffness.clone()
    # V_boundary_stiffness_threshold[V_boundary_stiffness_threshold < 1e-3] = 0
    V_boundary_stiffness_threshold = V_boundary_stiffness_threshold * torch.sigmoid(1e5 * (V_boundary_stiffness_threshold - 1e-3))
    dist_C = __get_spring_boundary_constraints_batch(softbody,
                                                      V_predict,
                                                      V_boundary_stiffness_threshold,
                                                      mask)
    # energy is C^2 * stiffness / 2
    boundary_energy = torch.square(dist_C) * V_boundary_stiffness_threshold / 2
    # print(boundary_energy.shape)
    return boundary_energy

def __get_spring_boundary_constraints_batch(softbody, V_predict, V_boundary_stiffness, mask=None):
    C = []
    C_stiffness = []
    for C_dist, C_init_d in zip(softbody.C_boundary_list, softbody.C_init_boundary_d_list):
        if mask == None or (C_dist[:, 0] in mask and C_dist[:, 1] in mask):
            # position difference vectors
            N = V_predict[:, C_dist[:, 0]] - V_predict[:, C_dist[:, 1]]
            # distance
            D = torch.norm(N, p=2, dim=-1, keepdim=True)
            # constarint values
            C.append(D - C_init_d)
            # average stiffness
            # C_stiffness.append(V_boundary_stiffness[C_dist[:, 0]])
    return torch.cat(C) #, torch.cat(C_stiffness)