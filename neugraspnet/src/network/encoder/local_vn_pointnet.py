import torch
import torch.nn as nn
import torch.nn.functional as F
from neugraspnet.src.network.conv_onet.models.utils.layers_equi import *
from torch_scatter import scatter_mean, scatter_max
from neugraspnet.src.network.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from neugraspnet.src.network.encoder.unet import UNet
from neugraspnet.src.network.encoder.unet3d import UNet3D

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class LocalPool_VN_Pointnet(nn.Module):
    ''' VectorNeuron-PointNet encoder (point-wise) with projection to local plane/grid features
        VectorNeuron layers provide a rotation-invariant feature representation
        Number of input points are fixed.
        Points get mean pooled locally to build the grid/plane features
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        n_knn (int): number of neighbors for input knn graph projection
        n_blocks (int): number of blocks ResNetBlockFC layers
        use_bnorm (bool): weather to use batch normalization
        scatter_type (str): feature aggregation when doing pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, n_knn=10, n_blocks=5, use_bnorm=True,
                 scatter_type='mean', unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=None, grid_resolution=None, plane_type=['xz', 'xy', 'yz'], padding=0.1):
        super(LocalPool_VN_Pointnet, self).__init__()
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.n_knn = n_knn
        
        self.conv_pos = VNLinearLeakyReLU(3, hidden_dim, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(hidden_dim, hidden_dim)
        self.block_0 = VNResnetBlockFC(hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(hidden_dim, c_dim)
        # self.fc_c = VNLinear(hidden_dim, c_dim)

        # self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool

        # conversion to invariant point-wise scalar features
        self.inv_std_feature = VNStdFeature(c_dim, dim=4, normalize_frame=True, use_batchnorm=False)

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'mean':
            self.scatter = scatter_mean
        elif scatter_type == 'max':
            self.scatter = scatter_max
        else:
            raise ValueError('incorrect scatter type')
    
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = self.scatter(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = self.scatter(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid


    def forward(self, p):
        batch_size, N, D = p.size()

        x = p.clone().transpose(1, 2) # pn needs B x D x N
        
        x = x.unsqueeze(1) # Add dimensions for 3D vectors
        feat = get_graph_feature_cross(x, k=self.n_knn)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)

        net = self.fc_pos(net)

        net = self.block_0(net)
        # Don't pool because this will result in global features
        # pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        net = self.block_2(net)
        net = self.block_3(net)
        net = self.block_4(net)

        # conversion to invariant point-wise scalar features
        x_inv, _ = self.inv_std_feature(net) # invariant vector
        x_inv = (x_inv * x_inv).sum(2) # norm to get scalar

        # latent features per-point
        c = x_inv.transpose(1, 2) # B x N x C

        # project features to planes/grid
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea