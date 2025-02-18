import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

DATASETS = ['fractal20220817_data', 'kuka', 'bridge', 'taco_play', 'jaco_play', 'berkeley_cable_routing', 'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 'toto', 'language_table', 'columbia_cairlab_pusht_real', 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds', 'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'cmu_franka_exploration_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds', 'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'qut_dexterous_manipulation', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_playing_with_food', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son', 'robot_vqa', 'droid', 'conq_hose_manipulation', 'dobbe', 'fmb', 'io_ai_tech', 'mimic_play', 'aloha_mobile', 'robo_set', 'tidybot', 'vima_converted_externally_to_rlds', 'spoc', 'plex_robosuite']


def dataset2path(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  elif dataset_name == 'droid':
     version=''
  else:
    version = '0.1.0'

  return f'gs://gresearch/robotics/{dataset_name}/{version}'


def as_gif(images, path='temp.gif'):
  # Render the images as the gif:
  images[0].save(path, save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open(path,'rb').read()
  return gif_bytes


def get_dataset(dataset,display_key='image'):
    b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
    print(b.info.features)

    if display_key not in b.info.features['steps']['observation']:
        raise ValueError(
            f"The key {display_key} was not found in this dataset.\n"
            + "Please choose a different image key to display for this dataset.\n"
            + "Here is the observation spec:\n"
            + str(b.info.features['steps']['observation']))

    ds = b.as_dataset(split='train[:10]').shuffle(10)   # take only first 10 episodes
    iterator = iter(ds)
    return ds, iterator

def get_features(dataset):
    b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
    return b.info.features

def get_image_from_episode(episode, display_key='image'):
    images = [step['observation'][display_key] for step in episode['steps']]
    images = [Image.fromarray(image.numpy()) for image in images]
    return images