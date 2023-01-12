
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.camera.ops import py_camera_model_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import camera_segmentation_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
2022.08.16 김대호
Waymo Open Dataset tutorial 참조 
'''


def distortion_correction(frame):
    w = frame.context.camera_calibrations[0].width
    h =  frame.context.camera_calibrations[0].height
    f_x, f_y, c_u, c_v, k_1, k_2, p_1, p_2, k_3= frame.context.camera_calibrations[0].intrinsic 

    mtx = np.array([[f_x, 0, c_u],[0, f_y, c_v],[0, 0, 1]])
    dist = np.array([[k_1, k_2, p_1, p_2, k_3]])
    print('1-1. Calibration: intrinsic matrix')
    print (mtx)
    print('1-2. Calibration: distortion coefficients')
    print (dist)

    plt.figure(figsize=(18, 10))

    frame_jpg = tf.image.decode_jpeg(frame.images[1].image)

    plt.subplot(1, 2, 1)
    plt.imshow (frame_jpg)
    plt.title('distortion')
    plt.axis = ('off')

    frame_jpg = np.array(frame_jpg)
    undist, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    img_undist = cv2.undistort(frame_jpg, mtx, dist, None, undist)

    plt.subplot(1, 2, 2)
    plt.imshow(img_undist)
    plt.title('undistortion')
    plt.axis = ('off')
    
def _pad_to_common_shape(image):
            return np.pad(image, [[1280 - image.shape[0], 0], [0, 0], [0, 0]])
    
def origianl_images_to_panorama(frames_unordered):
    camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                                open_dataset.CameraName.FRONT_LEFT,
                                open_dataset.CameraName.FRONT,  
                                open_dataset.CameraName.FRONT_RIGHT,
                                open_dataset.CameraName.SIDE_RIGHT] 
                              
    frames_ordered = []

    for frame in frames_unordered:
        image_proto_dict = {image.name : image.image for image in frame.images}
        frames_ordered.append([image_proto_dict[name] for name in camera_left_to_right_order])

    images_decode = [[tf.image.decode_jpeg(frame) for frame in frames ] for frames in frames_ordered]
    padding_images = [[_pad_to_common_shape(image) for image in images ] for images in images_decode]
    panorama_image_no_concat = [np.concatenate(image, axis=1) for image in padding_images]
    panorama_image = np.concatenate(panorama_image_no_concat, axis=0)

    plt.figure(figsize=(64, 60))
    plt.imshow(panorama_image)
    plt.grid(False)
    plt.axis('off')
    plt.show()
    
def show_image_with_label(frame, data, name, layout, cmap=None):
    plt.figure(figsize=(25, 20))
    ax = plt.subplot(*layout)
    for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != data.name:
            continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:
        # Draw the object bounding box.
            ax.add_patch(patches.Rectangle(
            xy=(label.box.center_x - 0.5 * label.box.length,
                label.box.center_y - 0.5 * label.box.width),
            width=label.box.length,
            height=label.box.width,
            linewidth=1,
            edgecolor='red',
            facecolor='none')) 

  # Show the camera image.
    plt.imshow(tf.image.decode_jpeg(data.image), cmap=cmap)
    plt.title(open_dataset.CameraName.Name.Name(data.name))
    plt.grid(False)
    #plt.axis('off')

def show_frame_image_with_label(frame):
    for index, image in enumerate(frame.images):
        show_image_with_label(frame, image, frame.camera_labels, [3, 3, index+1])

def show_image(camera_image, layout):
    """Display the given camera image."""
    plt.figure(figsize=(25,20))
    ax = plt.subplot(*layout)
    plt.imshow(tf.image.decode_jpeg(camera_image.image))
    plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    #plt.axis('off')
    return ax

def show_frame_image(frame):
    for index, image in enumerate(frame.images):
        _ = show_image(image, [3, 3, index + 1])

def convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False):
  """Convert range images from polar coordinates to Cartesian coordinates.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
      will be 3 if keep_polar_features is False (x, y, z) and 6 if
      keep_polar_features is True (range, intensity, elongation, x, y, z).
  """
  cartesian_range_images = {}
  frame_pose = tf.convert_to_tensor(
      value=np.reshape(np.array(frame.pose.transform), [4, 4]))

  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(value=range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)

  for c in frame.context.laser_calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == dataset_pb2.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

    if keep_polar_features:
      # If we want to keep the polar coordinate features of range, intensity,
      # and elongation, concatenate them to be the initial dimensions of the
      # returned Cartesian range image.
      range_image_cartesian = tf.concat(
          [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

    cartesian_range_images[c.name] = range_image_cartesian

  return cartesian_range_images

def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
  """Convert range images to point cloud.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []
  cp_points = []

  cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    range_image_cartesian = cartesian_range_images[c.name]
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][ri_index]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor,
                                    tf.compat.v1.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())

  return points, cp_points

def float_3d_point(frame, range_images, camera_projections, range_image_top_pose):
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                                                       range_images,
                                                       camera_projections,
                                                       range_image_top_pose)
    points = np.concatenate(points, axis=0)
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')

    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]

    ax.scatter3D(X,Y,Z,s=0.15)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    axes_limit = 50
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_zlim(-axes_limit, axes_limit)

    plt.show()


def draw_3d_wireframe_box(ax, u, v, color, linewidth=3):
  """Draws 3D wireframe bounding boxes onto the given axis."""
  # List of lines to interconnect. Allows for various forms of connectivity.
  # Four lines each describe bottom face, top face and vertical connectors.
  lines = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7))

  for (point_idx1, point_idx2) in lines:
    line = plt.Line2D(
        xdata=(int(u[point_idx1]), int(u[point_idx2])),
        ydata=(int(v[point_idx1]), int(v[point_idx2])),
        linewidth=linewidth,
        color=list(color) + [0.5])  # Add alpha for opacity
    ax.add_line(line)

def project_vehicle_to_image(vehicle_pose, calibration, points):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ],
                         dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()


def show_projected_camera_synced_boxes(frame, camera_image, ax):
    """Displays camera_synced_box 3D labels projected onto camera."""
    # Fetch matching camera calibration.
    calibration = next(cc for cc in frame.context.camera_calibrations if cc.name == camera_image.name)

    for label in frame.laser_labels:
        box = label.camera_synced_box

        if not box.ByteSize():
            continue 

        # Retrieve upright 3D box corners.
        box_coords = np.array([[
            box.center_x, box.center_y, box.center_z, box.length, box.width,
            box.height, box.heading
        ]])
        corners = box_utils.get_upright_3d_box_corners(
            box_coords)[0].numpy()  # [8, 3]

        # Project box corners from vehicle coordinates onto the image.
        projected_corners = project_vehicle_to_image(frame.pose, calibration,
                                                 corners)
        u, v, ok = projected_corners.transpose()
        ok = ok.astype(bool)

        # Skip object if any corner projection failed. Note that this is very
        # strict and can lead to exclusion of some partially visible objects.
        if not all(ok):
            continue
        u = u[ok]
        v = v[ok]

        # Clip box to image bounds.
        u = np.clip(u, 0, calibration.width)
        v = np.clip(v, 0, calibration.height)

        if u.max() - u.min() == 0 or v.max() - v.min() == 0:
            continue


        # Draw approximate 3D wireframe box onto the image. Occlusions are not
        # handled properly.
        draw_3d_wireframe_box(ax, u, v, (1.0, 1.0, 0.0))

def show_frame_image_with_3d_label(frame):
    plt.figure(figsize=(25, 20))
    for index, image in enumerate(frame.images):
        ax = show_image(image, [3, 3, index + 1])
        show_projected_camera_synced_boxes(frame, image, ax)
    plt.show()

def show_panorama_image_with_panoptic_label(frames_with_seg):
    camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                                open_dataset.CameraName.FRONT_LEFT,
                                open_dataset.CameraName.FRONT,
                                open_dataset.CameraName.FRONT_RIGHT,
                                open_dataset.CameraName.SIDE_RIGHT]
    segmentation_protos_ordered = []

    for frame in frames_with_seg:
        segmentation_proto_dict = {image.name : image.camera_segmentation_label for image in frame.images}
        segmentation_protos_ordered.append([segmentation_proto_dict[name] for name in camera_left_to_right_order])

    segmentation_protos_flat = sum(segmentation_protos_ordered, [])
    panoptic_labels, is_tracked_masks, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_protos(
        segmentation_protos_flat, remap_values=True
    )

    NUM_CAMERA_FRAMES = 5
    semantic_labels_multiframe = []
    instance_labels_multiframe = []

    for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
        semantic_labels = []
        instance_labels = []
        for j in range(NUM_CAMERA_FRAMES):
            semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                panoptic_labels[i + j], panoptic_label_divisor)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)

    semantic_labels_multiframe.append(semantic_labels)
    instance_labels_multiframe.append(instance_labels)

    instance_labels = [[_pad_to_common_shape(label) for label in instance_labels] for instance_labels in instance_labels_multiframe]
    semantic_labels = [[_pad_to_common_shape(label) for label in semantic_labels] for semantic_labels in semantic_labels_multiframe]
    instance_labels = [np.concatenate(label, axis=1) for label in instance_labels]
    semantic_labels = [np.concatenate(label, axis=1) for label in semantic_labels]

    instance_label_concat = np.concatenate(instance_labels, axis=0)
    semantic_label_concat = np.concatenate(semantic_labels, axis=0)
    panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
        semantic_label_concat, instance_label_concat)

    plt.figure(figsize=(64, 60))
    plt.imshow(panoptic_label_rgb)
    plt.grid(False)
    #plt.axis('off')
    plt.show()
    
  
