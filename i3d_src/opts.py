import argparse

parser = argparse.ArgumentParser('I3D: inflated inception v1 network)')

# test arguments
parser.add_argument(
    '--test', action='store_true', help='run validation mode')

# RGB arguments
parser.add_argument(
    '--rgb', action='store_true', help='Evaluate RGB pretrained network')

parser.add_argument(
    '--resume', type=str, default='', help='path to latest checkpoint')

parser.add_argument(
    '--rgb_weights_path',
    type=str,
    default='model/model_rgb.pth',
    help='Path to rgb model state_dict')

parser.add_argument(
    '--rgb_sample_path',
    type=str,
    default='data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy',
    help='Path to kinetics rgb numpy sample')

# Flow arguments
"""
parser.add_argument(
    '--flow', action='store_true', help='Evaluate flow pretrained network')
parser.add_argument(
    '--flow_weights_path',
    type=str,
    default='model/model_flow.pth',
    help='Path to flow model state_dict')
parser.add_argument(
    '--flow_sample_path',
    type=str,
    default='data/kinetic-samples/v_CricketShot_g04_c01_flow.npy',
    help='Path to kinetics flow numpy sample')
"""
# Class argument
parser.add_argument(
    '--classes_path',
    type=str,
    default='data/classes.txt',
    help='Path of the file containing classes names')
"""
# Sample arguments

parser.add_argument(
    '--sample_num',
    type=int,
    default='16',
    help='The number of the output frames after the sample, or 1/sample_rate frames will be chosen.')

parser.add_argument(
    '--out_fps',
    type=int,
    default='5',
    help='The fps of the output video.')

parser.add_argument(
    '--sample_type',
    type=str,
    default='fps',
    help="'fps': sample the video to a certain FPS, or 'num': control the number of output video, "
         "choose the video sample method.")

# Predict argument
parser.add_argument(
    '--top_k',
    type=int,
    default='3',
    help='When display_samples, number of top classes to display')

parser.add_argument(
    '--sample_path',
    type=str,
    help='Path of the sample video or image set you want to use to predict.')
"""
# Other argument
parser.add_argument(
    '--session_id',
    type=str,
    default='hmdb51_dataset',
    help='The session id of this training process.')

parser.add_argument(
    '--out_frame_num',
    type=int,
    default='32',
    help='The number of frames sent to model finally.')
