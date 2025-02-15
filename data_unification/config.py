CTRL_FREQ = {
    'RT-1 Robot Action': 3,
    'QT-Opt': 10,
    'Berkeley Bridge': 5,
    'Freiburg Franka Play': 15,
    'USC Jaco Play': 10,
    'Berkeley Cable Routing': 10,
    'Roboturk': 10,
    'NYU VINN': 3,
    'Austin VIOLA': 20,
    'Berkeley Autolab UR5': 5,
    'TOTO Benchmark': 30,
    'Language Table': 10,
    'Columbia PushT Dataset': 10,
    'Stanford Kuka Multimodal': 20,
    'NYU ROT': 3,
    'Stanford HYDRA': 10,
    'Austin BUDS': 20,
    'NYU Franka Play': 3,
    'Maniskill': 20,
    'Furniture Bench': 10,
    'CMU Franka Exploration': 10,
    'UCSD Kitchen': 2,
    'UCSD Pick Place': 3,
    'Austin Sailor': 20,
    'Austin Sirius': 20,
    'BC-Z': 10,
    'USC Cloth Sim': 10,
    'Tokyo PR2 Fridge Opening': 10,
    'Tokyo PR2 Tabletop Manipulation': 10,
    'Saytap': 50,
    'UTokyo xArm PickPlace': 10,
    'UTokyo xArm Bimanual': 10,
    'Robonet': 1,
    'Berkeley MVP Data': 5,
    'Berkeley RPT Data': 30,
    'KAIST Nonprehensile Objects': 10,
    'QUT Dynamic Grasping': 30,
    'Stanford MaskVIT Data': None,
    'LSMO Dataset': 10,
    'DLR Sara Pour Dataset': 10,
    'DLR Sara Grid Clamp Dataset': 10,
    'DLR Wheelchair Shared Control': 5,
    'ASU TableTop Manipulation': 13,
    'Stanford Robocook': 5,
    'ETH Agent Affordances': 67,
    'Imperial Wrist Cam': 10,
    'CMU Franka Pick-Insert Data': 20,
    'QUT Dexterous Manpulation': 30,
    'MPI Muscular Proprioception': 500,
    'UIUC D3Field': 1,
    'Austin Mutex': 20,
    'Berkeley Fanuc Manipulation': 10,
    'CMU Food Manipulation': 10,
    'CMU Play Fusion': 5,
    'CMU Stretch': 10,
    'RECON': 3,
    'CoryHall': 5,
    'SACSoN': 10,
    'RoboVQA': 10,
    'ALOHA': 50,
    'DROID': 15,
    'ConqHose': 30,
    'DobbE': None,
    'FMB': 10,
    'IO-AI Office PicknPlace': 30,
    'MimicPlay': 15,
    'MobileALOHA': 50,
    'RoboSet': 5,
    'TidyBot': None,
    'VIMA': None
}

DATA_CONFIG = {
    'RT-1 Robot Action': {
        'EEF_POS': {
            'key': 'base_pose_tool_reached',
            'start': None,
            'end': None
        },
        'JOINT_POS': None,
        'GRIPPER_POS': {
            'key': 'gripper_closed',
            'start': None,
            'end': None
        },
        'data_format': 'quaternion',
    },
    'QT-Opt': {
        'EEF_POS': {
            'key': 'clip_function_input/base_pose_tool_reached',
            'start': None,
            'end': None
        },
        'JOINT_POS': None,
        'GRIPPER_POS': {
            'key': 'gripper_closed',
            'start': None,
            'end': None
        },
    },
    'Berkeley Bridge': {
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': 6
        },
        'JOINT_POS': None,
        'GRIPPER_POS': {
            'key': 'state',
            'start': 6,
            'end': None
        },
        'data_format': 'rpy',
    },
    'Freiburg Franka Play': {
        'EEF_POS': {
            'key': 'robot_obs',
            'start': None,
            'end': 6
        },
        'JOINT_POS': {
            'key': 'robot_obs',
            'start': 7,
            'end': 14
        },
        'GRIPPER_POS': {
            'key': 'robot_obs',
            'start': 14,
            'end': None
        },
        'data_format': 'rpy',
    },
    'USC Jaco Play': {
        'EEF_POS': {
            'key': 'end_effector_cartesian_pos',
            'start': None,
            'end': 6
        },
        'JOINT_POS': {
            'key': 'joint_pos',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': None,
        'data_format': 'quaternion',
    },
    'Berkeley Cable Routing': {
        'EEF_POS': {
            'key': 'robot_state',
            'start': None,
            'end': None
        },
        'JOINT_POS': None,
        'GRIPPER_POS': None,
        'data_format': 'quaternion',
    },
    'Roboturk': None,
    'NYU VINN': None,
    'Austin VIOLA': {
        'EEF_POS': {
            'key': 'ee_states',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'joint_states',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'gripper_states',
            'start': None,
            'end': None
        },
        'data_format': 'homogeneous_matrix',
    },
    'Berkeley Autolab UR5': {
        'EEF_POS': {
            'key': 'robot_state',
            'start': 6,
            'end': 13
        },
        'JOINT_POS': {
            'key': 'robot_state',
            'start': None,
            'end': 6
        },
        'GRIPPER_POS': {
            'key': 'robot_state',
            'start': 13,
            'end': 14
        },
        'data_format': 'quaternion',
    },
    'TOTO Benchmark': {
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': None,
        'data_format': None,
    },
    'Language Table': None,
    'Columbia PushT Dataset': {
        'EEF_POS': {
            'key': 'robot_state',
            'start': None,
            'end': None
        },
        'JOINT_POS': None,
        'GRIPPER_POS': None,
        'data_format': 'xy',
    },
    'Stanford Kuka Multimodal': { # TODO: Need exception handling
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'joint_pos',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'gripper_closed',
            'start': None,
            'end': None
        },
        'data_format': None,
    },
    'NYU ROT': None,
    'Stanford HYDRA': {
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'JOINT_POS': {
            'key': 'state',
            'start': 10,
            'end': 17
        },
        'GRIPPER_POS': {
            'key': 'gripper_closed',
            'start': 26,
            'end': None
        },
        'data_format': 'quaternion',
    },
    'Austin BUDS': {
        'EEF_POS': {
            'key': 'state',
            'start': 8,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': {
            'key': 'state',
            'start': 7,
            'end': 8
        },
        'data_format': 'homogeneous_matrix',
    },
    'NYU Franka Play': {
        'EEF_POS': {
            'key': 'state',
            'start': 7,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': None,
        'data_format': 'rpy',
    },
    'Maniskill': {
        'EEF_POS': {
            'key': 'tcp_pose',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': {
            'key': 'state',
            'start': 7,
            'end': 8
        },
        'data_format': 'quaternion',
    },
    'Furniture Bench': {
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'JOINT_POS': {
            'key': 'state',
            'start': 13,
            'end': 20
        },
        'GRIPPER_POS': {
            'key': 'state',
            'start': 34,
            'end': None
        },
        'data_format': 'shifted_quaternion',
    },
    'CMU Franka Exploration': None,
    'UCSD Kitchen': { # TODO: Need Action field
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': None,
        'data_format': None,
    },
    'UCSD Pick Place': {
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': 6
        },
        'JOINT_POS': None,
        'GRIPPER_POS': {
            'key': 'state',
            'start': 6,
            'end': None
        },
        'data_format': 'rpy',
    },
    'Austin Sailor': {
        'EEF_POS': {
            'key': 'state_ee',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state_joint',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'state_gripper',
            'start': None,
            'end': None
        },
        'data_format': 'homogeneous_matrix',
    },
    'Austin Sirius': {
        'EEF_POS': {
            'key': 'state_ee',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state_joint',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'state_gripper',
            'start': None,
            'end': None
        },
        'data_format': 'homogeneous_matrix',
    },
    'BC-Z': { # TODO: Need exception handling
        'EEF_POS': {
            'key': 'present/xyz', # 'rot in present/axis_angle'
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': None,
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'present/sensed_close',
            'start': None,
            'end': None
        },
        'data_format': None,
    },
    'USC Cloth Sim': None,
    'Tokyo PR2 Fridge Opening': None,
    'Tokyo PR2 Tabletop Manipulation': None,
    'Saytap': None,
    'UTokyo xArm PickPlace': {
        'EEF_POS': {
            'key': 'end_effector_pose',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'joint_state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': None,
        'data_format': 'rpy',
    },
    'UTokyo xArm Bimanual': None,
    'Robonet': { # TODO: Need exception handling
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': 4
        },
        'JOINT_POS': None,
        'GRIPPER_POS': {
            'key': 'state',
            'start': 4,
            'end': None
        },
        'data_format': 'yaw',
    },
    'Berkeley MVP Data': {
        'EEF_POS': {
            'key': 'pose',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'joint_pos',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'gripper',
            'start': None,
            'end': None
        },
        'data_format': 'quaternion',
    },
    'Berkeley RPT Data': {
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'joint_pos',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'gripper',
            'start': None,
            'end': None
        },
        'data_format': None,
    },
    'KAIST Nonprehensile Objects': {
        'EEF_POS': {
            'key': 'state',
            'start': 14,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': None,
        'data_format': 'shifted_quaternion',
    },
    'QUT Dynamic Grasping': None,
    'Stanford MaskVIT Data': { # TODO: Need rpy convertion
        'EEF_POS': {
            'key': 'end_effector_pose',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'states',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': {
            'key': 'states',
            'start': -1,
            'end': None
        },
        'data_format': 'yaw',
    },
    'LSMO Dataset': None,
    'DLR Sara Pour Dataset': None,
    'DLR Sara Grid Clamp Dataset': None,
    'DLR Wheelchair Shared Control': None,
    'ASU TableTop Manipulation': {  # TODO: Need exception handling
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 6
        },
        'GRIPPER_POS': {
            'key': 'state',
            'start': -1,
            'end': None
        },
        'data_format': None,
    },
    'Stanford Robocook': {
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': 6
        },
        'JOINT_POS': None,
        'GRIPPER_POS': {
            'key': 'state',
            'start': -1,
            'end': None
        },
        'data_format': 'rpy',
    },
    'ETH Agent Affordances': {  # TODO: Abnormal simulation
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': 6
        },
        'JOINT_POS': None,
        'GRIPPER_POS': {
            'key': 'state',
            'start': -2,
            'end': -1
        },
        'data_format': 'rpy',
    },
    'Imperial Wrist Cam': None,
    'CMU Franka Pick-Insert Data': { # TODO: Need Action field
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': {
            'key': 'state',
            'start': 7,
            'end': 8
        },
        'data_format': None,
    },
    'QUT Dexterous Manpulation': None,
    'MPI Muscular Proprioception': None,
    'UIUC D3Field': {
        'EEF_POS': {
            'key': 'state',
            'start': None,
            'end': None
        },
        'JOINT_POS': None,
        'GRIPPER_POS': None,
        'data_format': 'homogeneous_matrix',
    },
    'Austin Mutex': {
        'EEF_POS': {
            'key': 'state',
            'start': 8,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': {
            'key': 'state',
            'start': 7,
            'end': 8
        },
        'data_format': 'homogeneous_matrix',
    },
    'Berkeley Fanuc Manipulation': None,
    'CMU Food Manipulation': None,
    'CMU Play Fusion': {
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': 7
        },
        'GRIPPER_POS': {
            'key': 'state',
            'start': 7,
            'end': 8
        },
        'data_format': None,
    },
    'CMU Stretch': None,
    'RECON': None,
    'CoryHall': None,
    'SACSoN': None,
    'RoboVQA': None,
    'ALOHA': None,
    'DROID': {
        'EEF_POS': {
            'key': 'cartesian_position',
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'joint_position',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'gripper_position',
            'start': None,
            'end': None
        },
        'data_format': 'rpy',
    },
    'ConqHose': { # Joint position abnormal
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'state',
            'start': 20,
            'end': 40
        },
        'GRIPPER_POS': None,
        'data_format': None,
    },
    'DobbE': None,
    'FMB': None,
    'IO-AI Office PicknPlace': None,
    'MimicPlay': { # TODO: Need exception handling
        'EEF_POS': {
            'key': 'state', # ee_pose
            'start': None,
            'end': None
        },
        'JOINT_POS': {
            'key': 'state', # joint_positions
            'start': None,
            'end': None
        },
        'GRIPPER_POS': {
            'key': 'state', # gripper_position
            'start': None,
            'end': None
        },
        'data_format': 'quaternion',
    },
    'MobileALOHA': {
        'EEF_POS': None,
        'JOINT_POS': {
            'key': 'state',
            'start': None,
            'end': None
        },
        'GRIPPER_POS': None,
        'data_format': None,
    },
    'RoboSet': None,
    'TidyBot': None,
    'VIMA': None,
}

env2xml = {
    'RT-1 Robot Action': 'mujoco_menagerie/google_robot/robot.xml',
    'QT-Opt': 'mujoco_menagerie/google_robot/robot.xml',
    'Berkeley Bridge': 'mujoco_menagerie/berkeley_bridge/bridge.xml',
    'Freiburg Franka Play': 'mujoco_menagerie/franka_fr3/fr3.xml',
    'USC Jaco Play': 'mujoco_menagerie/jaco_arm/jaco_arm.xml',
    'Berkeley Cable Routing': 'mujoco_menagerie/berkeley_cable_routing/cable_routing.xml',
    
}