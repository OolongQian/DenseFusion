import argparse
import copy
import os

import cv2
import numpy as np
import numpy.ma as ma
import open3d as o3d
import scipy.io as scio
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='datasets/ycb/YCB_Video_Dataset', help='dataset root dir')
parser.add_argument('--model', type=str, default='trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth',
                    help='resume PoseNet model')
parser.add_argument('--refine_model', type=str,
                    default='trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth',
                    help='resume PoseRefineNet model')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'

id2name = {1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box', 4: '005_tomato_soup_can',
           5: '006_mustard_bottle', 6: '007_tuna_fish_can', 7: '008_pudding_box', 8: '009_gelatin_box',
           9: '010_potted_meat_can', 10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
           13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block', 17: '037_scissors',
           18: '040_large_marker', 19: '051_large_clamp', 20: '052_extra_large_clamp', 21: '061_foam_brick'}


def get_bbox(posecnn_rois):
	rmin = int(posecnn_rois[idx][3]) + 1
	rmax = int(posecnn_rois[idx][5]) - 1
	cmin = int(posecnn_rois[idx][2]) + 1
	cmax = int(posecnn_rois[idx][4]) - 1
	r_b = rmax - rmin
	for tt in range(len(border_list)):
		if r_b > border_list[tt] and r_b < border_list[tt + 1]:
			r_b = border_list[tt + 1]
			break
	c_b = cmax - cmin
	for tt in range(len(border_list)):
		if c_b > border_list[tt] and c_b < border_list[tt + 1]:
			c_b = border_list[tt + 1]
			break
	center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
	rmin = center[0] - int(r_b / 2)
	rmax = center[0] + int(r_b / 2)
	cmin = center[1] - int(c_b / 2)
	cmax = center[1] + int(c_b / 2)
	if rmin < 0:
		delt = -rmin
		rmin = 0
		rmax += delt
	if cmin < 0:
		delt = -cmin
		cmin = 0
		cmax += delt
	if rmax > img_width:
		delt = rmax - img_width
		rmax = img_width
		rmin -= delt
	if cmax > img_length:
		delt = cmax - img_length
		cmax = img_length
		cmin -= delt
	return rmin, rmax, cmin, cmax


estimator = PoseNet(num_points=num_points, num_obj=num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
	input_line = input_file.readline()
	if not input_line:
		break
	if input_line[-1:] == '\n':
		input_line = input_line[:-1]
	testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
	class_input = class_file.readline()
	if not class_input:
		break
	class_input = class_input[:-1]
	
	input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
	cld[class_id] = []
	while 1:
		input_line = input_file.readline()
		if not input_line:
			break
		input_line = input_line[:-1]
		input_line = input_line.split(' ')
		cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
	input_file.close()
	cld[class_id] = np.array(cld[class_id])
	class_id += 1

for now in range(0, 2949, 100):
	img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
	depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
	posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
	label = np.array(posecnn_meta['labels'])
	posecnn_rois = np.array(posecnn_meta['rois'])
	
	lst = posecnn_rois[:, 1:2].flatten()
	my_result_wo_refine = []
	my_result = []
	
	for idx in range(len(lst)):
		"this itemid means model class"
		itemid = lst[idx]
		
		try:
			rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)
			
			mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
			mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
			mask = mask_label * mask_depth
			
			choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
			try:
				if len(choose) > num_points:
					c_mask = np.zeros(len(choose), dtype=int)
					c_mask[:num_points] = 1
					np.random.shuffle(c_mask)
					choose = choose[c_mask.nonzero()]
				else:
					choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
			except ValueError:
				print("np.pad('wrap') error")
				continue
			
			depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
			xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
			ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
			choose = np.array([choose])
			
			pt2 = depth_masked / cam_scale
			pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
			pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
			cloud = np.concatenate((pt0, pt1, pt2), axis=1)
			
			img_masked = np.array(img)[:, :, :3]
			img_masked = np.transpose(img_masked, (2, 0, 1))
			img_masked = img_masked[:, rmin:rmax, cmin:cmax]
			
			cloud = torch.from_numpy(cloud.astype(np.float32))
			choose = torch.LongTensor(choose.astype(np.int32))
			img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
			index = torch.LongTensor([itemid - 1])
			
			cloud = Variable(cloud).cuda()
			choose = Variable(choose).cuda()
			img_masked = Variable(img_masked).cuda()
			index = Variable(index).cuda()
			
			cloud = cloud.view(1, num_points, 3)
			img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
			
			pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
			pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
			
			pred_c = pred_c.view(bs, num_points)
			how_max, which_max = torch.max(pred_c, 1)
			pred_t = pred_t.view(bs * num_points, 1, 3)
			points = cloud.view(bs * num_points, 1, 3)
			
			my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
			my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
			my_pred = np.append(my_r, my_t)
			my_result_wo_refine.append(my_pred.tolist())
			
			for ite in range(0, iteration):
				T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points,
				                                                                                 1).contiguous().view(1,
				                                                                                                      num_points,
				                                                                                                      3)
				my_mat = quaternion_matrix(my_r)
				R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
				my_mat[0:3, 3] = my_t
				
				new_cloud = torch.bmm((cloud - T), R).contiguous()
				pred_r, pred_t = refiner(new_cloud, emb, index)
				pred_r = pred_r.view(1, 1, -1)
				pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
				my_r_2 = pred_r.view(-1).cpu().data.numpy()
				my_t_2 = pred_t.view(-1).cpu().data.numpy()
				my_mat_2 = quaternion_matrix(my_r_2)
				
				my_mat_2[0:3, 3] = my_t_2
				
				my_mat_final = np.dot(my_mat, my_mat_2)
				my_r_final = copy.deepcopy(my_mat_final)
				my_r_final[0:3, 3] = 0
				my_r_final = quaternion_from_matrix(my_r_final, True)
				my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
				
				my_pred = np.append(my_r_final, my_t_final)
				my_r = my_r_final
				my_t = my_t_final
			
			# Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
			
			my_result.append(my_pred.tolist())
			
			"here we finally have both class item id and predicted poses."
			clsid = itemid
			assert my_pred is not None
			pose = my_pred.tolist()
			
			pt2 = depth_masked / cam_scale
			pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
			pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
			cloud = np.concatenate((pt0, pt1, pt2), axis=1)
			
			img_masked = np.array(img)[:, :, :3]
			img_masked = np.transpose(img_masked, (2, 0, 1))
			img_masked = img_masked[:, rmin:rmax, cmin:cmax]
			
			"visualize masked rgb image"
			# print("class id {} name {}".format(clsid, id2name[clsid]))
			# plt.imshow(img_masked_display)
			# plt.show()
			
			"visualize masked rgbd point cloud"
			# pcd = o3d.geometry.PointCloud()
			# pcd.points = o3d.utility.Vector3dVector(cloud)
			
			"store rgb, depth, clouds, instance id, class id."
			result_images_dir = 'experiments/eval_result/ycb/masked_images'
			# result_depths_dir = 'experiments/eval_result/ycb/masked_depths'
			result_plds_dir = 'experiments/eval_result/ycb/masked_plds'
			
			img_masked = np.transpose(img_masked, (1, 2, 0))
			cv2.imwrite(os.path.join(result_images_dir,
			                         '_'.join(testlist[now].split('/')) + "_index_{}_classid_{}.png".format(idx, int(clsid))),
			            img_masked)
			
			# cv2.imwrite(os.path.join(result_depths_dir,
			#                          testlist[now] + "_index_{}_classname_{}.png".format(idx, id2name[clsid])),
			# 	depth_masked)
			
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(cloud)
			o3d.io.write_point_cloud(
				os.path.join(result_plds_dir, '_'.join(testlist[now].split('/')) + "_index_{}_classid_{}.ply".format(idx, int(clsid))),
				pcd)
		
		except ZeroDivisionError:
			print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
			my_result_wo_refine.append([0.0 for i in range(7)])
			my_result.append([0.0 for i in range(7)])
	
	# scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '_'.join(testlist[now].split('/'))),
	#              {'poses': my_result_wo_refine})
	# scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '_'.join(testlist[now].split('/'))), {'poses': my_result})
	print("Finish No.{0} keyframe".format(now))
