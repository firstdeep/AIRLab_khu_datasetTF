import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import camera_segmentation_utils
from utils import *

FILE_NAME_1 = 'set file path'
FILE_NAME_2 = 'set file path'
if __name__ == '__main__' : 
    #Read one frame
    dataset = tf.data.TFRecordDataset(FILE_NAME_1, compression_type='')
    
    frames_unordered = []

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames_unordered.append(frame)
        break

    (range_images, camera_projections,_, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame))
    
    distortion_correction(frame)
    origianl_images_to_panorama(frames_unordered)
    show_frame_image_with_label(frame)
    float_3d_point(frame, range_images, camera_projections, range_image_top_pose)
    
    #Read segmentation label frame
    dataset = tf.data.TFRecordDataset(FILE_NAME_2, compression_type='')
    frames_with_seg = []

    sequence_id = None

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if frame.images[0].camera_segmentation_label.panoptic_label:
            frames_with_seg.append(frame)
            frames_unordered.append(frame)
            if sequence_id is None:
                sequence_id = frame.images[0].camera_segmentation_label.sequence_id
            if frame.images[0].camera_segmentation_label.sequence_id != sequence_id or len(frames_with_seg) > 2:
                break

    show_frame_image_with_3d_label(frame)
    show_panorama_image_with_panoptic_label(frames_with_seg)
