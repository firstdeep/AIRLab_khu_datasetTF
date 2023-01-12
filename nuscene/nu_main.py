from nu_lib import *
from utils import *

'''
2022.07.16 황병훈 
Nuscene github 참조.
refactoring 진행 중...
'''

if __name__ == "__main__":

    img_w = 1600
    img_h = 900

    '''
    Camera Field of View 
    2ArcTan(sensor width / 2 * focal length)) * (180/pi) 
    '''

    camera_list = ['CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    nusc = NuScenes(version='v1.0-mini', dataroot='../nuimage_sample', verbose=True)

    '''
    # Sample token list 
    # 'token' 'timestamp' 'prev' 'next' 'scene_token' 'data' 'anns'
    # data 안에는 해당 scene에 맞는 RADAR, CAMERA, CAM 의 token 존재
    '''
    sample = nusc.sample[0] # sample 할당

    scene_record = nusc.get('scene', sample['scene_token'])
    total_num_samples = scene_record['nbr_samples'] # key frame True 인 것만 count (lidar & camera matching 시 사용)
    first_sample_token = scene_record['first_sample_token']
    last_sample_token = scene_record['last_sample_token']
    current_token = first_sample_token
    current_sample = nusc.get('sample', current_token)

    # timestamp 관련 변수 (단위: nanosecond)
    first_timestamp = nusc.get('sample', scene_record['first_sample_token'])['timestamp']
    start_time = nusc.get('sample', scene_record['first_sample_token'])['timestamp'] / 1000000
    length_time = nusc.get('sample', scene_record['last_sample_token'])['timestamp'] / 1000000 - start_time

    '''
    # Sample_data list 
    # 'token' 'sample_token' 'ego_pose_token' 'calibrated_sensor_token' 'timestamp' 'fileformat' is_key_frame' 
    # 'height' 'width' 'filename' 'prev' 'next' 'sensor_modality' 'channel' 
    '''
    curr_sample_data = []
    for idx, i in enumerate(camera_list):
        curr_sample_data.append(nusc.get('sample_data', sample['data'][i]))


    nu = NuSceneExplorer(nusc)

    lidar_token = sample['data']['LIDAR_TOP']
    cam_token = sample['data']['CAM_FRONT']
    cam_sample = nusc.get("sample_data", cam_token)
    cam_intrinsic = nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])['camera_intrinsic']

    fov_x, fov_y = cal_fov(img_w, img_h, cam_intrinsic)

    # # Information of 6 camera
    # # ego info, camera parameter, file_path load
    # sample_info = load_info(nusc, curr_sample_data)
    #
    # # 카메라 intrinsic 대부분 비슷함 (CAM_BACK 제외)
    # ego_ex, camera_ex, camera_in = load_camera_coord(nusc, sample_info)

    """ Visualization """
    # nu.lidar_points_visualization(sample=sample)

    ''' Projection to image '''
    nu.lidar_projection_to_image(cam_token=cam_token, lidar_token=lidar_token, dst_path= './lidar_to_image_result.jpg')
