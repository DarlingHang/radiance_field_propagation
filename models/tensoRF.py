from .tensorBase import *

class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.fg_plane, self.fg_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.bg_plane, self.bg_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.sem_plane, self.sem_line = self.init_one_svd(self.sem_n_comp, self.gridSize, 0.1, device)
        self.basis_mat_fg = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        self.basis_mat_bg = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        self.basis_mat_sem = torch.nn.Linear(sum(self.sem_n_comp), self.sem_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.fg_line, 'lr': lr_init_spatialxyz}, {'params': self.fg_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.bg_line, 'lr': lr_init_spatialxyz}, {'params': self.bg_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.sem_line, 'lr': lr_init_spatialxyz}, {'params': self.sem_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat_fg.parameters(), 'lr':lr_init_network}, {'params': self.basis_mat_bg.parameters(), 'lr':lr_init_network}, {'params': self.basis_mat_sem.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.fg_plane)):
            total = total + reg(self.fg_plane[idx]) * 1e-2 + reg(self.fg_line[idx]) * 1e-3
            total = total + reg(self.bg_plane[idx]) * 1e-2 + reg(self.bg_line[idx]) * 1e-3
        return total
    
    def TV_loss_sem(self, reg):
        total = 0
        for idx in range(len(self.sem_plane)):
            total = total + reg(self.sem_plane[idx]) * 1e-2 + reg(self.sem_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point_fg,line_coef_point_fg = [],[]
        for idx_plane in range(len(self.fg_plane)):
            plane_coef_point_fg.append(F.grid_sample(self.fg_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point_fg.append(F.grid_sample(self.fg_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point_fg, line_coef_point_fg = torch.cat(plane_coef_point_fg), torch.cat(line_coef_point_fg)

        plane_coef_point_bg,line_coef_point_bg = [],[]
        for idx_plane in range(len(self.bg_plane)):
            plane_coef_point_bg.append(F.grid_sample(self.bg_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point_bg.append(F.grid_sample(self.bg_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point_bg, line_coef_point_bg = torch.cat(plane_coef_point_bg), torch.cat(line_coef_point_bg)


        return self.basis_mat_fg((plane_coef_point_fg * line_coef_point_fg).T), self.basis_mat_bg((plane_coef_point_bg * line_coef_point_bg).T)

    
    def compute_semfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.sem_plane)):
            plane_coef_point.append(F.grid_sample(self.sem_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.sem_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat_sem((plane_coef_point * line_coef_point).T)

