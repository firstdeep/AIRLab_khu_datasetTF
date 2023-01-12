import glob
import natsort
import numpy as np
from nuscenes import NuScenes
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from camera_models import *
import PIL.Image
import copy
from sympy import Line3D, Point3D, Segment3D

"""
# ego pose 란 
# ego의 좌표는 timestamp와 연관이 있음 따라서 timestamp가 다르면 같은 장면이라도 slightly 하게 차이날 수 있음 
# 차량이 계속 움직이기 때문에 ego는 계속 변함 
# ego의 기준은 world frame 
# world 는 좌측 상단의 좌표 값 기준 (총 4개?) 
"""

class NuSceneExplorer:
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

        self.show_lidarseg = True
        self.show_panoptic = False
        self.flat_vehicle = True


    def load_info(self, sample):

        camera_list = ['CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK']

        curr_sample_data = []
        for idx, i in enumerate(camera_list):
            curr_sample_data.append(self.nusc.get('sample_data', sample['data'][i]))

        sample_info = []
        for frame in curr_sample_data:
            sample_dict = {"ego": [], "camera_info": [], 'file_path': []}

            # ego frame
            ego_frame = self.nusc.get("ego_pose", frame['ego_pose_token'])

            # camera calibration
            camera_info = self.nusc.get("calibrated_sensor", frame['calibrated_sensor_token'])

            # file_name
            file_path = frame['filename']

            sample_dict['ego'].append(ego_frame)
            sample_dict['camera_info'].append(camera_info)
            sample_dict['file_path'].append(file_path)

            sample_info.append(sample_dict)

        return sample_info

    def lidar_to_ego(self, sample):

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_recoder = self.nusc.get('sample_data', lidar_token)

        # Ensure that lidar pointcloud is from a keyframe.
        assert lidar_recoder['is_key_frame'], \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborte'

        pcl_path = os.path.join('../nuimage_sample', lidar_recoder['filename'])
        pc = LidarPointCloud.from_file(pcl_path)  # Shape: 4*34688 (top에서 본 값)
        # point cloud는 individual 하며 unrelated 하다.
        # 왜 shape이 4 인가? X,Y,Z + intensity

        if self.flat_vehicle:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = self.nusc.get('calibrated_sensor', lidar_recoder['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', lidar_recoder['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record["rotation"]), inverse=False)

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]  # yaw:x, pitch:y, roll:x

            # Quaternion으로 변경하는 것의 의미
            # Quarternion과 lidar yaw 축과의 관계
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        axes_limit = 50

        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

        semantic_table = getattr(self.nusc, 'lidarseg')

        return points, colors


    def lidar_points_visualization(self, sample):
        draw_axis = True
        scatt = True
        axes_limit = 15
        point_scale = 0.2

        cam = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc_path = os.path.join('../nuimage_sample', lidar['filename'])

        # point cloud file open
        if lidar['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(pc_path)
            origin_pc = LidarPointCloud.from_file(pc_path)

        """
        @주의점@
        camera -> ego -> global 로 가는 상황에서는 
        roation 먼저 곱한 후에 tranlation을 더하는 것으로 진행 
        
        하지만 반대의 경우 
        global -> ego -> camera 의 경우 
        translation을 먼저 빼준 후, rotation의 inverse(transpose)를 곱해주면 된다. 
        """
        lidar_cam = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(lidar_cam['rotation']).rotation_matrix)  # Rotation matrix shape: 3 * 3
        pc.translate(np.array(lidar_cam['translation']))  # point cloud의 X,Y,Z 각각에 대해서 더해준다.

        # Second step: transform from ego to the global frame.
        lidar_ego = self.nusc.get('ego_pose', lidar['ego_pose_token'])
        pc.rotate(Quaternion(lidar_ego['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_ego['translation']))

        camera_ego = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(camera_ego['translation']))
        pc.rotate(Quaternion(camera_ego['rotation']).rotation_matrix.T)

        camera_cam = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(camera_cam['translation']))
        pc.rotate(Quaternion(camera_cam['rotation']).rotation_matrix.T)


        fig = plt.figure(figsize=(9, 9))
        ax = plt.axes(projection="3d")

        x = pc.points[0,:]
        y = pc.points[1,:]
        z = pc.points[2,:]

        ax.scatter3D(pc.points[0,:], pc.points[1,:], pc.points[2,:], alpha=0.5, s=point_scale, color='b')
        ax.scatter3D(origin_pc.points[0,:], origin_pc.points[1,:], origin_pc.points[2,:], alpha=0.5, s=point_scale, color='r')


        # Show ego vehicle.
        ax.plot(0, 0, 0, 'x', color='red')

        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
        ax.set_zlim(-axes_limit, axes_limit)

        # ax.axis('off')

        plt.show()


    '''
    lidar point 를 image coordinate에 projectin하는 function 
    좌표 변환 
    lidar coordinate --> lidar ego --> global --> camera extrinsic --> camera intrinsic --> image coordinate
    '''
    def lidar_projection_to_image(self, cam_token, lidar_token, dst_path):
        # render lidar intensity instead of point depth.
        cam = self.nusc.get('sample_data', cam_token)
        lidar = self.nusc.get('sample_data', lidar_token)
        point_path = os.path.join('../nuimage_sample', lidar['filename'])

        if lidar['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(point_path)
            pc_copy = copy.deepcopy(pc)
            print(f"Point cloud shape: {pc.points.shape}")

        img = cv2.imread(os.path.join('../nuimage_sample', cam['filename']))
        im = PIL.Image.open(os.path.join('../nuimage_sample', cam['filename']))

        '''
        좌표변환 순서 
        1. extrinsic 사용 하여 point와 dot 해준다. (lidar -> ego vehicle 기준으로 변경) 
        2. ego pose의 extrinsic 사용하여 global frame 으로 변경 (lidar ego -> global) 
        3. global 에서 camera ego로 변경 (global -> camera ego) / translation 음수, rotation은 T곱하기 
        4. camera ego 에서 extrinsic 으로 좌표 변경 (camera ego -> camera coordinate) / translation 음수, rotation은 T곱하기 
        5. 최종적으로 image plane에 투영하기 위해서 intrinsic 곱해준다. 
        '''
        # coordinate 변환 중요!!!
        # Points live in the point sensor frame. so they need to be transformed via global to the image plane
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        # lidar 의 extrinsic 을 이용하여,
        lidar_cam = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(lidar_cam['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_cam['translation']))

        # Second step: transform from ego to the global frame.
        lidar_ego = self.nusc.get('ego_pose', lidar['ego_pose_token'])
        pc.rotate(Quaternion(lidar_ego['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_ego['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        camera_ego = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(camera_ego['translation']))
        pc.rotate(Quaternion(camera_ego['rotation']).rotation_matrix.T) # Transpose 곱셈의 의미

        # Fourth step: transform from ego into the camera.
        camera_cam = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(camera_cam['translation']))
        pc.rotate(Quaternion(camera_cam['rotation']).rotation_matrix.T) # R의 역행렬은 transpose 행렬이다.

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :] # 4 * ~ 인데 왜 2부터 시작하는지....?
        # 2면 Z축을 의미하고 Z축은 결국 depth 이다.

        rander_intensity = True
        if rander_intensity:
            # Retrieve the color from the intensities.
            # Performs arbitary scaling to achieve more visually pleasing results.
            intensities = pc.points[3, :] # intensities 는 3 부터
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities)) # normalization
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5) # 0 ~ 0.5 사이로 맞춰준다.
            coloring = intensities # 근적외선의 반사 강도를 가지고 있는다. (0~255) 사시의 값을 가지고 있다.
            # 낮은 숫자는 낮은 반사율을 나타내고 높은 숫자는 높은 반사율을 나타낸다.

        else:
            coloring = depths

        ego_yaw = True

        if ego_yaw:
            cs_record = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record['rotation']))
            ego_yaw = Quaternion(lidar_ego['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(lidar_ego['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

            points_lidar = view_points(pc_copy.points[:3, :], viewpoint, normalize=False)

            for i in range(3):
                points_lidar[i,:] = points_lidar[i,:] - camera_cam['translation'][i]
            points_lidar = np.dot(Quaternion(camera_cam['rotation']).rotation_matrix.T, points_lidar)

            points_lidar = view_points(points_lidar, np.array(camera_cam['camera_intrinsic']), normalize=True)

        # points = view_points(pc.points[:3, :], np.array(camera_cam['camera_intrinsic']), normalize=True)
        points = points_lidar

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1M in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        # 일단 point cloud는 360도로 존재함, 따라서 image plane에 mapping 하기 위해서는 카메라 intrinsic 을 가져와서 projection 시켜야 함
        # 카메라 뒤에 있는 점은 모두 지우고, 또한 카메라 1m앞에 있는 점만 찾는다.
        # sync는 안맞을 수 있음
        mask = np.ones(depths.shape[0], dtype=bool)
        min_dist = 1.0 # (1M)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0,:]>1)
        mask = np.logical_and(mask, points[0,:] < img.shape[1]-1)
        mask = np.logical_and(mask, points[1,:]>1)
        mask = np.logical_and(mask, points[1,:] < img.shape[0]-1)
        points = points[:, mask]
        coloring = coloring[mask]

        dot_size = 2
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.imshow(im)
        ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
        ax.axis('off')
        plt.show()
        plt.savefig(dst_path)

        # return points, coloring, img


    def load_camera_coord(self, sample):
        ego_extrinsic = []
        camera_extrinsic = []
        camera_intrinsic = []

        for frame in sample:
            ego_r = frame['ego'][0]['rotation']
            ego_r_matrix = Quaternion(ego_r).rotation_matrix
            ego_t = np.array(frame['ego'][0]['translation']).reshape(3, 1)
            ego_extrinsic.append(np.concatenate((ego_r_matrix, ego_t), axis=1))

            camera_r = frame['camera_param'][0]['rotation']  # Quaternion rotation 4*1
            camera_r_matrix = Quaternion(camera_r).rotation_matrix
            camera_t = np.array(frame['camera_param'][0]['translation']).reshape(3, 1)

            camera_intrinsic.append(frame['camera_param'][0]['camera_intrinsic'])
            camera_extrinsic.append(np.concatenate((camera_r_matrix, camera_t), axis=1))

        return ego_extrinsic, camera_extrinsic, camera_intrinsic


    def write_timestamp(self, sample, file_name='timestamp_log.txt'):
        f = open(file_name, 'w')
        lidar_sample = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        timestamp = lidar_sample['timestamp']
        key_frame = lidar_sample['is_key_frame']
        f.write(str(timestamp) + '\n')
        count = 0
        total = 0

        while (True):
            if lidar_sample['next'] == '':
                break
            lidar_next = self.nusc.get('sample_data', lidar_sample['next'])
            timestamp = lidar_next['timestamp']
            key_frame = lidar_next['is_key_frame']
            if key_frame:
                count = count + 1
            f.write(str(timestamp) + '\n')
            lidar_sample = lidar_next
            total = total + 1
            print(key_frame)
        f.close()



    def next_sample(self, curr_sample_data):
        next_sample_data = []

        for curr in curr_sample_data:
            # if curr['next'] =='':
            #     next_sample_data.append(None)
            # else:
            next_sample_data.append(self.nusc.get('sample_data', curr['next']))

        return next_sample_data

