
from datasets import build_dataset, get_coco_api_from_dataset
import argparse
import numpy as np
import os
from PIL import Image
import random


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file',default='config/DINO_self_training/DINO_4scale.py', type=str, required=False)
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--masks', default=False)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


dataset_train_strong_aug = build_dataset(image_set='train', args=args, strong_aug=True)




if __name__ == "__main__":

    source_img_dir = '' #输入存放图像文件夹的路径

    select_proportion = 0.05  #选择百分比

    all_files = os.listdir(source_img_dir)
    # 随机选取 5%
    num_to_select = int(len(all_files) * select_proportion)
    random.seed(42)  # 确保每次结果一致
    selected_samples = random.sample(all_files, num_to_select)
    print(f"随机选取的{select_proportion * 100}%文件：共{num_to_select}个")


    # 从 selected_samples 找到对应的 image_id
    selected_image_ids = []

    # 遍历 COCO 中的所有图像信息，建立 file_name 到 image_id 的反查字典
    file_name_to_id = {v['file_name']: k for k, v in dataset_train_strong_aug.coco.imgs.items()}

    # 匹配选中的文件名，获取对应的 image_id
    for file_name in selected_samples:
        if file_name in file_name_to_id:
            selected_image_ids.append(file_name_to_id[file_name])
        else:
            print(f"警告：文件名 {file_name} 不存在于 COCO 图像信息中，将跳过！")

    # # 可选：打印获取到的 image_id 数量和示例
    print(f"成功匹配到 {len(selected_image_ids)} 个 image_id：")


    #
    save_np_path = os.path.join("selected_indices_random5_SSDD.npy")
    np.save(save_np_path, selected_image_ids, allow_pickle=True)