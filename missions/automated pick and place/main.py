import pybullet as p
import pybullet_data

import util
from util import move_to_joint_pos, gripper_open, gripper_close

import matplotlib.pyplot as plt
import numpy as np

import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from external.ggcnn.candidate_grasps import grasp_predictor

def move_to_ee_pose(robot_id, ee_link_id, target_ee_pos, target_ee_orientation=None):
    """
    Moves the robot to a given end-effector pose.
    :param robot_id: pyBullet's body id of the robot
    :param target_ee_pos: (3,) list/ndarray with target end-effector position
    :param target_ee_orientation: (4,) list/ndarray with target end-effector orientation as quaternion
    """
    inverse_kinematic = p.calculateInverseKinematics(
        robot_id,
        ee_link_id,
        targetPosition = target_ee_pos,
        targetOrientation = target_ee_orientation,
        maxNumIterations = 100,
        residualThreshold = 0.001
    )

    joint_pos = inverse_kinematic[:7]

    move_to_joint_pos(robot_id, joint_pos)

def forward_kinematic(robot_id, ee_link_id): 
    """
    Computes position and orientation of end-effector
    :param robot_id: pyBullet's body id of the robot
    :param ee_link_id: pyBullet's end-effector link id
    """
    pos, quat, *_ = p.getLinkState(robot_id, ee_link_id, computeForwardKinematics=True)
    print(f"End-Effector pose @ config:, position {pos}, orientation {quat}")

def get_ee_camera_view(robot_id, ee_link_id):
    """
    Obtains camera view and projection matrices for robot's end-effector
    :param robot_id: pyBullet's body id of the robot
    :param ee_link_id: pyBullet's end-effector link id
    """
    state = p.getLinkState(robot_id, ee_link_id, computeForwardKinematics=True)
    ee_pos = state[0]
    ee_orn = state[1]

    rot_matrix = p.getMatrixFromQuaternion(ee_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    forward = rot_matrix @ np.array([0, 0, 1]) 
    up = rot_matrix @ np.array([0, 1, 0])       

    camera_target = ee_pos + 0.1 * forward
    view_matrix = p.computeViewMatrix(cameraEyePosition=ee_pos, cameraTargetPosition=camera_target, cameraUpVector=up)
    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=300/300, nearVal=0.01, farVal=2.0)
    return view_matrix, projection_matrix

def get_camera_image(view_matrix, projection_matrix, width=300, height=300):
    """
    Using the provided view matrix and projection matrix, this function captures the image
    :param view_matrix: camera's position and orientation in the simulation
    :param projection_matrix: representation of 3D scene in a 2D plane
    :param width: width of the capturing image
    :param height: height of the capturing image
    """
    img_arr = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=projection_matrix, 
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

    rgb_img = np.reshape(img_arr[2], (height, width, 4))[:, :, :3] 
    depth_buffer = np.reshape(img_arr[3], (height, width))

    return rgb_img, depth_buffer 

def show_depthimage(depth):
    """
    Displays the calculated depth image
    :param depth: returned depth buffer from get_camera_image function
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(depth, cmap='gray')
    plt.axis('off')
    plt.show()

def show_rgbimage(rgb): 
    """
    Displays the calculated rgb image
    :param depth: returned rgb image from get_camera_image function
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

def convert_coordinates(x, y, depth_buffer, view_matrix, projection_matrix, width=300, height=300):
    """
    Using the provided image co-ordinates, this function converts them into workspace co-ordinates
    :param x: calculated x of grasp centre
    :param y: calculated y of grasp centre
    :param depth_buffer: calculated depth at calculated grasp centre
    :param view_matrix: camera's position and orientation in the simulation
    :param projection_matrix: representation of 3D scene in a 2D plane
    :param width: width of the capturing image
    :param height: height of the capturing image
    """
    # Depth value at proposed grasp centre
    buffer = depth_buffer[y, x]
    
    # Normalising pixel co-ordinate to normalised device co-ordinates, range of -1 to +1
    normalised_x = (x / width - 0.5) * 2.0
    normalised_y = (0.5 - y / height) * 2.0
    normalised_z = 2.0 * buffer - 1.0 

    # Building normalised co-ordinates
    normalised_coordinates = np.array([normalised_x, normalised_y, normalised_z, 1.0])

    # Ensuring shape (4,4) and then transposing
    projection_matrix = np.array(projection_matrix).reshape(4,4).T
    view_matrix = np.array(view_matrix).reshape(4,4).T

    # Reversing from image to world, retracing steps
    inverse_projection = np.linalg.inv(projection_matrix)
    inverse_view = np.linalg.inv(view_matrix)

    # Unprojecting normalised co-ordinates to 3D space
    view_coordinates = inverse_projection @ normalised_coordinates
    view_coordinates /= view_coordinates[3]

    # Reversing camera transformation with inverse view and view co-ordinates
    world_coords = inverse_view @ view_coordinates
    world_coords /= world_coords[3]

    return world_coords[:3]

def grasp_success(object_id, tray_id): 
    """
    Using built-in position retrieval code, this function can determine whether a grasp was successful
    The function works by comparing the position of the cube in relation to the destination tray, to validate whether it was placed successfully 
    :param object_id: pyBullet id for the cube
    :param tray_id: pyBullet id for the tray
    """
    # Retrieving position of cube in simulation 
    cube_position, _ = p.getBasePositionAndOrientation(object_id)
    cube_x, cube_y, _ = cube_position

    # Retrieving position of tray in simulation 
    tray_position, _ = p.getBasePositionAndOrientation(tray_id)
    tray_x, tray_y, _ = tray_position

    # Tolerance which is deemed acceptable
    x_acceptance, y_acceptance = 0.2, 0.18

    # Calculating minimum and maximum values for x and y
    x_min, x_max = tray_x - x_acceptance, tray_x + x_acceptance 
    y_min, y_max = tray_y - y_acceptance, tray_y + y_acceptance 

    # Finding if cube location intersects tray location
    if x_min <= cube_x <= x_max and y_min <= cube_y <= y_max: 
        return 'Success, cube is in tray'
    else: 
        return 'Failed, cube is not in tray'
     
def generate_depth(robot_id): 
    """
    This function allows the depth image to be saved, needed for grasp visualisation and running automated grasps
    :param robot_id: pyBullet's body id of the robot
    """
    # pre-grasp 
    print('going above cube')
    move_to_ee_pose(robot_id, util.ROBOT_EE_LINK_ID, [0.5, -0.3, 0.225])
    gripper_open(robot_id)

    # camera view at pre-grasp
    view_matrix, proj_matrix = get_ee_camera_view(robot_id, util.ROBOT_EE_LINK_ID)
    rgb_img, depth_img = get_camera_image(view_matrix, proj_matrix)
    plt.imsave('images/cube_small.png', depth_img, cmap='gray')

    print('Image saved successfully')

def automate_grasps(robot_id, object_id, tray_id): 
    """
    Allows the automation of graps bypassing the GUI, using DIRECT mode
    :param robot_id: pyBullet's body id of the robot
    :param object_id: pyBullet id for the cube
    :param tray_id: pyBullet id for the tray
    """
    # pre-grasp 
    print('going above cube')
    move_to_ee_pose(robot_id, util.ROBOT_EE_LINK_ID, [0.5, -0.3, 0.225])
    gripper_open(robot_id)

    # camera view at pre-grasp
    view_matrix, proj_matrix = get_ee_camera_view(robot_id, util.ROBOT_EE_LINK_ID)
    rgb_img, depth_img = get_camera_image(view_matrix, proj_matrix)

    # calling function from candidate_graps.py
    depth_image_path = 'images/cube_small.png'
    x, y, angle_radians = grasp_predictor(depth_image_path)

    # plugging predicted values into converter
    world_cords = convert_coordinates(x, y, depth_img, view_matrix, proj_matrix)
    gripper_orientation = [np.cos(angle_radians / 2), 0, 0, np.sin(angle_radians / 2)]

    # going to calculated location 
    print(f'going to calculated grasp centre {world_cords}')
    move_to_ee_pose(robot_id, util.ROBOT_EE_LINK_ID, world_cords, gripper_orientation)
    gripper_close(robot_id)

    # going to location of tray
    print('going to tray location')
    move_to_ee_pose(robot_id, util.ROBOT_EE_LINK_ID, [0.5, 0.5, 0.3])
    gripper_open(robot_id)

    # displaying location of cube
    cube_position, _ = p.getBasePositionAndOrientation(object_id)
    print(f'Final Cube Position: {cube_position}')

    # call to validation function, to measure grasp success
    is_success = grasp_success(object_id, tray_id)
    print(f'Result -- {is_success}')

    # home config
    print('going to home configuration')
    move_to_joint_pos(robot_id, util.ROBOT_HOME_CONFIG)

    print('Grasp Completed!')

def main():
    # mode selection 
    print('Please select from the following options:')
    print('1 - Capture Depth Image & Exit')
    print('2 - Run Automated Grasp')
    mode = int(input('Please enter 1 or 2: '))

    # connect to pybullet in direct mode
    p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.81) # setting gravity
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # allows us to load plane, robots, etc.
    plane_id = p.loadURDF('plane.urdf')  # function returns an ID for the loaded body
    robot_id = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True) # load the robot

    # load an object to be grasped
    object_id = p.loadURDF('cube_small.urdf', basePosition=[0.5, -0.3, 0.025], baseOrientation=[0, 0, 0, 1])
    p.resetVisualShapeData(object_id, -1, rgbaColor=[1, 0, 0, 1])

    # loading tray for object to be placed
    tray_id = p.loadURDF('tray/traybox.urdf', basePosition=[0.5, 0.5, 0.0], baseOrientation=[0, 0, 0, 1])

    # home config
    print('going to home configuration')
    move_to_joint_pos(robot_id, util.ROBOT_HOME_CONFIG)

    # mode selection, can either save depth image or run the grasp directly
    if mode == 1: 
        generate_depth(robot_id)
    elif mode == 2:
        automate_grasps(robot_id, object_id, tray_id)
    else: 
        print('Please select from options 1 or 2')
    
    # clean up
    p.disconnect()

if __name__ == '__main__':
    main()