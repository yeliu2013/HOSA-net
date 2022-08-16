# Pytorch Implementation of  HOSA-Net: high-order statistic aggregation deep
network for 3D point clouds

This repo is based on the Pytorch implementation of PointNet and PointNet++:.

https://github.com/yanx27/Pointnet_Pointnet2_pytorch

# ModelNet40
python train_classification.py

# Part Segmentation (ShapeNet)
python train_partseg.py

Both pointnet and pointnet++ are supported.
 
Different configuration of HOSA module can be selected by modifying:

classifier = model.get_model(num_class, normal_channel=args.use_normals, use_hosa=True, code=['MAX', 'OM1', 'CM2', 'CM3', 'CM4'], use_sa=True, agg_type=0)

'code' is the high order statistics that are selected, statistic-wise attention is selected if 'use_sa'is True. Aggregator type is set by 'agg_type'  
