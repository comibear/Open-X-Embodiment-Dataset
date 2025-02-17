import tensorflow_datasets as tfds
import tensorflow as tf
import json
import numpy as np
import logging
from typing import Optional, Mapping, Sequence, Tuple, Union, Callable

from data_unification.src.data.dlimp.dataset import DLataset as dl
from data_unification.src.utils.spec import ModuleSpec
from data_unification.src.data.utils.data_utils import (
    NormalizationType,
    get_dataset_statistics,
    tree_map,
    normalize_action_and_proprio,
)
from functools import partial

from package.mujoco_usage.mujoco_parser import *
from package.helper.transformation import pR2xyzq, rpy2quat, r2quat, quat2r
from data_unification.src.data.xml_parser import parse_robot_structure
from data_unification.config import DATA_CONFIG, CTRL_FREQ, env2xml

log = logging.getLogger(__name__)

'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(8,), dtype=float32),
        'discount': Scalar(shape=(), dtype=float32),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'gripper': Scalar(shape=(), dtype=bool),
            'hand_image': Image(shape=(480, 640, 3), dtype=uint8),
            'joint_pos': Tensor(shape=(7,), dtype=float32),
            'pose': Tensor(shape=(7,), dtype=float32),
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})
'''
# TODO: 0. Adding language instruction
def get_language_instruction(traj: dict) -> str:
    return traj["language_instruction"]

# TODO: 1. Adding zero state (structure)
def get_zero_state(env: MuJoCoParserClass, exclude_links: Sequence[str] = ['world']) -> np.ndarray:
    env.reset()
    env.forward(q=np.zeros(env.n_qpos))
    
    _, body_names = parse_robot_structure(env.name)
    body_pos = [env.get_pR_body(body_name) for body_name in body_names]
    body_pos = [pR2xyzq(pos) for pos in body_pos]
    body_pos = np.concatenate(body_pos, axis=0)
    
    return body_pos

# TODO: 2. Adding end-effector pose
# TODO: 3. Adding joint positions
# TODO: 4. Adding Action (eef velocities)
def all2xyzq(data, format: str = 'quaternion') -> np.ndarray:
    if format == 'quaternion':
        return data
    elif format == 'rpy':
        return np.concatenate([data[:3], rpy2quat(data[3:])], axis=0)
    elif format == 'homogeneous_matrix':
        # Extract rotation matrix and position from 4x4 homogeneous matrix
        data = data.reshape(4, 4)
        pos = data[:3, 3]
        rot_mat = data[:3, :3]
        quat = r2quat(rot_mat)
        return np.concatenate([pos, quat], axis=0)
    elif format == 'shifted_quaternion':
        # Convert from xyzw to wxyz quaternion format
        pos = data[:3]
        quat = np.array([data[6], data[3], data[4], data[5]])
        return np.concatenate([pos, quat], axis=0)
    elif format == 'yaw':
        # Set roll and pitch to 0, keep yaw
        pos = data[:3]
        yaw = data[3]
        rpy = np.array([0.0, 0.0, yaw])
        return np.concatenate([pos, rpy2quat(rpy)], axis=0)
    else:
        raise ValueError(f"Invalid format: {format}")

def get_additional_data(env: MuJoCoParserClass, traj, traj_len: int) -> np.ndarray:
    env_name = env.name
    joint_names, body_names = parse_robot_structure(env_name)
    
    joint_pos_list = np.array([])
    eef_pos_list = np.array([])
    eef_vel_list = np.array([])
    
    env.reset()
    prev_eef_pos = None
    if DATA_CONFIG[env_name]['JOINT_POS']:
        key = DATA_CONFIG[env_name]['JOINT_POS']['key']
        j_s, j_e = DATA_CONFIG[env_name]['JOINT_POS']['start'], DATA_CONFIG[env_name]['JOINT_POS']['end']
        
        for step in tfds.as_numpy(traj["steps"]):
            jp = step["observation"][key][j_s:j_e]
            joint_pos_list = np.append(joint_pos_list, jp) # joint_pos
        
            env.forward(q=jp, joint_names=joint_names)
            eef_pos = pR2xyzq(env.get_pR_body(body_name='tcp_link'))
            eef_pos_list = np.append(eef_pos_list, eef_pos) # eef_pos
            
            if prev_eef_pos is not None:
                eef_vel = (eef_pos - prev_eef_pos) / CTRL_FREQ[env_name]
                eef_vel_list = np.append(eef_vel_list, eef_vel) # eef_vel
            
            prev_eef_pos = eef_pos
            
    elif DATA_CONFIG[env_name]['EEF_POSE']:
        key = DATA_CONFIG[env_name]['EEF_POSE']['key']
        e_s, e_e = DATA_CONFIG[env_name]['EEF_POSE']['start'], DATA_CONFIG[env_name]['EEF_POSE']['end']
        
        q0 = np.array(np.zeros(env.n_qpos)) # TODO: some environments requires specific initial joint configuration
        q_curr = q0.copy()
        for step in tfds.as_numpy(traj["steps"]):
            eef_pos = step["observation"][key][e_s:e_e]
            eef_pos = all2xyzq(eef_pos)
            eef_pos_list = np.append(eef_pos_list, eef_pos) # eef_pos
            
            if prev_eef_pos is not None:
                eef_vel = (eef_pos - prev_eef_pos) / CTRL_FREQ[env_name]
                eef_vel_list = np.append(eef_vel_list, eef_vel) # eef_vel
                
            prev_eef_pos = eef_pos
            
            # Joint positions
            
            # Get target end-effector pose
            p_target = eef_pos[:3]
            R_target = quat2r(eef_pos[3:])
            
            # Solve IK for current target
            qpos, ik_err_stack, _ = solve_ik(
                env = env,
                joint_names_for_ik = joint_names, 
                body_name_trgt = 'tcp_link',
                q_init = q_curr,
                p_trgt = p_target,
                R_trgt = R_target,
                max_ik_tick = 500,
                ik_stepsize = 1.0,
                ik_eps = 1e-2,
                ik_th = np.radians(5.0),
                render = False,
                verbose_warning = False
            )
            
            # Update joint positions if IK succeeded
            if np.abs(ik_err_stack).max() < 1e-2:
                q_curr = qpos.copy()
                joint_pos_list = np.append(joint_pos_list, qpos) # joint_pos
            else:
                q_curr = q0.copy()
                raise ValueError("IK failed")
            
            env.forward(q=qpos, joint_names=joint_names)
    
    return joint_pos_list, eef_pos_list, eef_vel_list

# TODO: 5. Exception Handler
def refine_exceptions(env_name: str) -> Callable:
    if env_name in DATA_CONFIG:
        return DATA_CONFIG[env_name]
    else:
        return None

def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    env_name: str,
    *,
    train: bool,
    split: Optional[str] = None,
    standardize_fn: Optional[ModuleSpec] = None,
    shuffle: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS,
    dataset_statistics: Optional[Union[dict, str]] = None,
    force_recompute_dataset_statistics: bool = False,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    filter_functions: Sequence[ModuleSpec] = (),
    skip_norm: bool = False,
    ignore_errors: bool = False,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict, int]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation" and "action". "observation"
    should be a dictionary containing some number of additional keys, which will be extracted into an even
    more standardized format according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in
    place of an old name to insert padding. For example, if after `standardize_fn`, your "observation" dict
    has RGB images called "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary":
    None, "wrist": "wrist"}`, then the resulting dataset will have an "observation" dict containing the keys
    "image_primary", "image_secondary", and "image_wrist", where "image_primary" corresponds to "workspace",
    "image_secondary" is a padding image, and "image_wrist" corresponds to "wrist".

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will
    contain the key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset,
            since one file usually contains many trajectories!).
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to
            extract from the "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in
            image_obs_keys.items()}`. If a value of `old` is None, inserts a padding image instead (empty
            string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        proprio_obs_key (str, optional): If provided, the "obs" dict will contain the key "proprio", extracted from
            `traj["observation"][proprio_obs_key]`.
        language_key (str, optional): If provided, the "task" dict will contain the key
            "language_instruction", extracted from `traj[language_key]`. If language_key fnmatches multiple
            keys, we sample one uniformly.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. May also provide "num_transitions" and "num_trajectories" keys for downstream usage
            (e.g., for `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        force_recompute_dataset_statistics (bool, optional): If True and `dataset_statistics` is None, will
            recompute the dataset statistics regardless of whether they are already cached.
        action_normalization_mask (Sequence[bool], optional): If provided, only normalizes action dimensions
            where the corresponding mask is True. For example, you might not want to normalize the gripper
            action dimension if it's always exactly 0 or 1. By default, all action dimensions are normalized.
        filter_functions (Sequence[ModuleSpec]): ModuleSpecs for filtering functions applied to the
            raw dataset.
        skip_norm (bool): If true, skips normalization of actions and proprio. Default: False.
        ignore_errors (bool): If true, skips erroneous dataset elements via dataset.ignore_errors(). Default: False.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    
    REQUIRED_KEYS = {"observation", "action"}

    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = ModuleSpec.instantiate(standardize_fn)(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
                "Did you write a `standardize_fn`?"
            )

        # Trajectory length
        traj_len = tf.shape(traj["action"])[0]

        old_obs = traj["observation"]
        new_obs = {}

        # Image observations
        for old_key in old_obs.keys():
            if 'image' in old_key:
                new_obs[old_key] = old_obs[old_key]
            elif 'depth' in old_key:
                new_obs[old_key] = old_obs[old_key]
            elif 'language' in old_key:
                new_obs[old_key] = old_obs[old_key]
                
        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)        
        
        # Environment additional data
        xml_path = env2xml[env_name]
        env = MuJoCoParserClass(rel_xml_path=xml_path,verbose=False)
        
        # TODO: Zero state
        env.reset()
        env.forward(q=np.zeros(env.n_qpos))
        
        # TODO: eef velocity
        
        # TODO: joint positions
        
        
        # TODO: eef position
        
        
        
        # Language instructions
        new_traj = {
            "observation": new_obs,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        return new_traj

    def is_nonzero_length(traj):
        return tf.shape(traj["action"])[0] > 0

    builder = tfds.builder(name, data_dir=data_dir)

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
            
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)
        for filter_fcn_spec in filter_functions:
            full_dataset = full_dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
        if ignore_errors:
            full_dataset = full_dataset.ignore_errors()
        full_dataset = full_dataset.traj_map(restructure).filter(is_nonzero_length)
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(proprio_obs_key),
                (
                    ModuleSpec.to_string(standardize_fn)
                    if standardize_fn is not None
                    else ""
                ),
                *map(ModuleSpec.to_string, filter_functions),
            ),
            save_dir=builder.data_dir,
            force_recompute=force_recompute_dataset_statistics,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if (
            len(action_normalization_mask)
            != dataset_statistics["action"]["mean"].shape[-1]
        ):
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # construct the dataset
    if split is None:
        if "val" not in builder.info.splits:
            split = "train[:95%]" if train else "train[95%:]"
        else:
            split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )
    dataset_len = len(dataset)

    # filter dataset
    for filter_fcn_spec in filter_functions:
        dataset = dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    if ignore_errors:
        dataset = dataset.ignore_errors()

    # organize data in dataset
    dataset = dataset.traj_map(restructure, num_parallel_calls).filter(is_nonzero_length)

    # normalization
    if not skip_norm:
        dataset = dataset.traj_map(
            partial(
                normalize_action_and_proprio,
                metadata=dataset_statistics,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls,
        )
    else:
        log.warning(
            "Dataset normalization turned off -- set skip_norm=False to apply normalization."
        )

    return dataset, dataset_statistics, dataset_len

