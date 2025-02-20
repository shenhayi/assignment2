from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            pass
            # TODO:
            # self.decoder =  
            self.layer0 = nn.Sequential(
                nn.Linear(512, 2048),
            )
            # layer2: Upsample the 3D volume using 3D transposed convolutions
            self.layer1 = nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 2->4
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # 4->8
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # 8->16
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),     # 16->32
                nn.Sigmoid()  # For occupancy probabilities
            )           
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # self.decoder =     
            self.layer0 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.n_point * 3)
            )        
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.num_vertices = self.mesh_pred.verts_packed().shape[0]
            # TODO:
            # self.decoder =       
            self.layer0 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_vertices * 3)
            )      

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred =   
            x = self.layer0(encoded_feat)
            x = x.reshape(-1, 256, 2, 2, 2)
            voxels_pred = self.layer1(x)        
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =    
            x = self.layer0(encoded_feat)
            pointclouds_pred = x.view(B, self.n_point, 3)        
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =      
            x = self.layer0(encoded_feat)      
            deform_vertices_pred = x.view(B, self.num_vertices, 3) 
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

