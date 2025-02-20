import torch

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	loss = torch.nn.functional.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	src_dist, _, _ = torch.ops.knn_points(point_cloud_src, point_cloud_tgt)
	tgt_dist, _, _ = torch.ops.knn_points(point_cloud_tgt, point_cloud_src)
	loss_chamfer = torch.mean(src_dist) + torch.mean(tgt_dist)
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = torch.loss.mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian
