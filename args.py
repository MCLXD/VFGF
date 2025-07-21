# coding=gbk
import argparse

def float_list(string):
    """将输入的字符串转换为浮点数列表"""
    return [float(value) for value in string.strip('[]').split(',')]
def parse_args():
    parser = argparse.ArgumentParser()

    # models
    parser.add_argument(
        '--model_name',
        default='base_srl',
        type=str,
        help='')
    parser.add_argument(
    '--best_model_name',
    default='',
    type=str,
    help='')
    parser.add_argument('--hidden', type=int, default=1024,
                        help='Number of hidden units')
    parser.add_argument('--feat_in', type=int, default=1024,
                        help='Input size. If fusion, it is discarded (see --feats_in)')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='The dimentation of semantic embedding.')

    # datasets
    parser.add_argument(
        '--dataset',
        default='EPIC',
        type=str,
        help='Used dataset (EPIC | 50Salads | Breakfast | Gaze)')
    parser.add_argument(
        '--modality',
        default='rgb',
        type=str, choices=['rgb', 'flow', 'obj', 'fusion'],
        help = "Modality. Rgb/flow/obj represent single branches, whereas fusion indicates the whole model with modality attention.")

    ## path
    parser.add_argument('--path_to_data',
                        default='data',
                        type=str, help="Path to the data folder, containing all LMDB datasets")
    parser.add_argument('--path_to_results', default='/mnt/data/Long_Datas/best_models/', type=str,
                        help='Result directory path')
    parser.add_argument('--resume_timestamp',
                        default='',
                        type=str,
                        help='resume timestamp')#default='./../../Datas/Gaze+/egtea/S1/   /mnt/data/Long_Datas/EPIC-55/
    parser.add_argument('--path_to_lmdb',
			default='../../../Datas/EPIC-100/',
			type=str,
			help='path to lmdb')
    parser.add_argument('--path_to_fusion_RGB',
                        default='results/R4.06.18',
                        type=str,
                        help="--path_to_fusion_RGB")
    parser.add_argument('--path_to_fusion_FLOW',
                        default='results/F4.12.00.3',
                        type=str,
                        help="--path_to_fusion_FLOW")#C:\mnt\data\Long_Datas\best_models\Gaze\base_srl\results
    parser.add_argument('--path_to_fusion_OBJ',
                        default='results/O4.08.02',
                        type=str,
                        help="--path_to_fusion_OBJ")
    parser.add_argument('--pre_train', action='store_true',
                        help='Whether to resume suspended training')
    ## train
    parser.add_argument(
        '--mode',
        default='train',
        type=str, choices=['train', 'validate', 'test'],
        help="Whether to perform training, validation or test. If test is selected, \
        --json_directory must be used to provide a directory in which to save the generated jsons.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--wd', type=float, default=5e-4, help="Weight decay")
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay')
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--ant_dropout', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--enc_dropout', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--clc_dropout', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--clip_value', default=1, type=float, help='grad clip value')
    parser.add_argument('--gpu_ids', default='7', type=str, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--milestones', '--arg', nargs='+', type=int, help='Weight Decay')
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of parallel thread to fetch the data")
    parser.add_argument('--reinforce_verb_weight', type=float, default=0.5, help="Reinforce module loss weigth")
    parser.add_argument('--reinforce_noun_weight', type=float, default=0.5, help="Reinforce module loss weigth")
    parser.add_argument('--revision_etp_weight', type=float, default=0.5, help="Revision NNL module loss weight")
    parser.add_argument('--revision_sd_weight', type=float, default=1.5, help="self-distillation module loss weight")
    parser.add_argument('--revision_ad_weight', type=float, default=3, help="Anti-distillation module loss weight")
    parser.add_argument('--revision_threshold', type=float_list, default=[0.003, 0.003, 0.003], help="Revision etp module data threshold")
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume suspended training')

    ## hyper
    parser.add_argument('--S_enc', type=int, default=6,
                        help="Number of encoding steps. \
                                If early recognition is performed, \
                                this value is discarded.")
    parser.add_argument('--S_ant', type=int, default=8,
                        help="Number of anticipation steps. \
                                If early recognition is performed, \
                                this is the number of frames sampled for each action.")
    parser.add_argument('--S_Rad', type=int, default=0,
                        help="Number of encoding steps. \
                                If early recognition is performed, \
                                this value is discarded.")
    parser.add_argument('--alpha', type=float, default=0.25,
                        help="Distance between time-steps in seconds")
    parser.add_argument('--task', type=str, default='anticipation', choices=[
                        'anticipation', 'early_recognition'], help='Task to tackle: anticipation or early recognition')
    parser.add_argument('--img_tmpl', type=str,
                        default='frame_{:010d}.jpg', help='Template to use to load the representation of a given frame')
    parser.add_argument('--display_every', type=int, default=10,
                        help="Display every n iterations")
    parser.add_argument('--Fusion_weight', type=list, default=[],
                        help="Weight of mixed modes")

    args = parser.parse_args()

    return args
