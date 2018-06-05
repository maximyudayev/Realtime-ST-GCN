#!/usr/bin/env python
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

class Demo(IO):
    """
        Demo for Skeleton-based Action Recgnition
    """
    def start(self):

        openpose = f'{self.arg.openpose}/examples/openpose/openpose.bin'
        video_name = self.arg.video.split('/')[-1].split('.')[0]
        output_snippets_dir = f'./data/openpose_estimation/snippets/{video_name}'
        output_sequence_dir = './data/openpose_estimation/data'
        output_sequence_path = f'{output_sequence_dir}/{video_name}.json'
        output_result_dir = self.arg.output_dir
        output_result_path = f'{output_result_dir}/{video_name}.avi'

        # pose estimation
        openpose_args = dict(
            video=self.arg.video,
            write_json=output_snippets_dir,
            display=0,
            render_pose=0)
        command_line = openpose + ' '
        command_line += ' '.join([f'--{k} {v}' for k, v in openpose_args.items()])
        shutil.rmtree(output_snippets_dir, ignore_errors=True)
        os.makedirs(output_snippets_dir)
        os.system(command_line)

        # pack openpose ouputs
        video = utils.video.get_video_frames(self.arg.video)
        height, width, _ = video[0].shape
        video_info = utils.openpose.json_pack(output_snippets_dir, video_name, width, height)
        if not os.path.exists(output_sequence_dir):
            os.makedirs(output_sequence_dir)
        with open(output_sequence_path, 'w') as outfile:
            json.dump(video_info, outfile)
        if len(video_info['data']) == 0:
            print('Can not find pose estimation results.')
            return
        else:
            print('Pose estimation complete.')

        # parse skeleton data
        pose, _ = utils.video.video_info_parsing(video_info)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev)

        # extract feature
        print('Network forwad.')
        output, feature = self.model.extract_feature(data)[0]
        intensity = feature.abs().sum(dim=0)
        intensity = intensity.cpu().detach().numpy()

        edge = self.model.graph.edge
        images = utils.visualization.stgcn_visualize(pose, edge, intensity, video)

        # save video
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        skvideo.io.vwrite(output_result_path, np.stack(images), outputdict={
                    '-b': '300000000'})
        print(f'The Demo result has been saved in {output_result_path}.')

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # openpose
        parser.add_argument('--video',
            default='./resource/media/Kop0sDqOn-c.mp4',
            help='Path to video')
        parser.add_argument('--openpose',
            default='3dparty/openpose/build',
            help='Path to openpose')
        parser.add_argument('--output_dir',
            default='./data/demo_result',
            help='Path to save results')
        parser.set_defaults(config='./config/st_gcn/kinetics_skeleton/demo.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser