import numpy as np


def read_skeleton(file):
    matx = np.loadtxt(file, delimiter=',')
    skeleton_sequence = {}
    skeleton_sequence['numFrame'] = matx.shape[0]
    # if skeleton_sequence['numFrame'] > 256:
    #     skeleton_sequence['numFrame'] = 256
    skeleton_sequence['frameInfo'] = []
    for t in range(skeleton_sequence['numFrame']):
        frame_info = {}
        frame_info['numBody'] = 1
        frame_info['bodyInfo'] = []
        for m in range(frame_info['numBody']):
            body_info = {}
            body_info['numJoint'] = matx.shape[1]
            body_info['jointInfo'] = []
            iterator = [0, 6, 12, 18, 24]
            #iterator = [0, 3, 6, 9, 12, 15]
            for v in iterator:
                joint_info_key = [
                    'x', 'y', 'z', 'x_g', 'y_g', 'z_g'
                ]
                joint_info = {
                    k: float(v)
                    for k, v in zip(joint_info_key, matx[t,[v,v+1,v+2,v+3,v+4,v+5]])
                }
                body_info['jointInfo'].append(joint_info)
            frame_info['bodyInfo'].append(body_info)
        skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=1, num_joint=9):
    seq_info = read_skeleton(file)
    data = np.zeros((6, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z'], v['x_g'], v['y_g'], v['z_g']]
                else:
                    pass
    return data