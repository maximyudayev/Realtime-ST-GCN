import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_confusion_matrix_rt():
    with open('pretrained_models/pku-mmdv1/realtime/train_9_64_50/confusion_matrix.csv','r') as f:
        confusion_matrix = np.genfromtxt(f, delimiter=',', dtype=np.int32)[1:,1:]

    plt.subplots()
    plt.imshow(confusion_matrix, cmap='magma', interpolation='nearest', norm=colors.LogNorm(vmin=1, vmax=confusion_matrix.max(), clip=True))
    # plt.imshow(confusion_matrix, cmap='magma', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_segmentation_masks():
    fig, axs = plt.subplots(3)

    trials = [37]
    # trials = [37, 175, 293]
    # letters = ['a)','b)','c)']
    paths = [
        'pretrained_models/pku-mmdv1/realtime/train_9_64_50',
        'pretrained_models/pku-mmdv1/realtime/train_21_64_50',
        'pretrained_models/pku-mmdv1/realtime/train_69_64_50',
        'pretrained_models/pku-mmdv1/realtime/train_153_64_50',
        'pretrained_models/pku-mmdv1/realtime/train_299_64_50',
        'pretrained_models/pku-mmdv1/original/train_9_50_64_50',
        'pretrained_models/pku-mmdv1/original/train_21_50_64_50',
        'pretrained_models/pku-mmdv1/original/train_69_50_64_50',]

    for i in range(len(trials)):
        with open('{0}/segmentation-{1}.csv'.format(paths[0],trials[i]),'r') as f:
            lines = f.read().splitlines()
        mask_prediction = np.array(lines[2].split(',')[1:], dtype=np.int32, ndmin=2)

        for j in range(1,len(paths)):
            with open('{0}/segmentation-{1}.csv'.format(paths[j],trials[i]),'r') as f:
                lines = f.read().splitlines()
            
            mask_prediction = np.concatenate((mask_prediction, np.array(lines[2].split(',')[1:], dtype=np.int32, ndmin=2)), axis=0)

        mask_label = np.array(lines[1].split(',')[1:], dtype=np.int32, ndmin=2)
        
        labels = [
            'Ground Truth',
            'RT-ST-GCN$_{\Gamma=9}$',
            'RT-ST-GCN$_{\Gamma=21}$',
            'RT-ST-GCN$_{\Gamma=69}$',
            'RT-ST-GCN$_{\Gamma=153}$',
            'RT-ST-GCN$_{\Gamma=299}$',
            'ST-GCN$_{L=50,\Gamma=9}$',
            'ST-GCN$_{L=50,\Gamma=21}$',
            'ST-GCN$_{L=50,\Gamma=69}$']

        axs[i].imshow(np.concatenate((mask_label, mask_prediction), axis=0), cmap='terrain_r', vmin=0.0, vmax=51.0, aspect='auto', interpolation='nearest')
        # axs[i].set_ylabel(letters[i], rotation=0, fontsize='xx-large', fontweight='bold')
        axs[i].set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], labels=labels, fontsize='x-large')
        axs[i].tick_params(length=0.0)
        axs[i].set_frame_on(False)

    fig.tight_layout()
    plt.show()


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


def plot_accuracy_gap():
    paths = [
        'pretrained_models/pku-mmdv1/realtime/train_9_64_50',
        # 'pretrained_models/pku-mmdv1/realtime/train_21_64_50',
        # 'pretrained_models/pku-mmdv1/realtime/train_69_64_50',
        # 'pretrained_models/pku-mmdv1/realtime/train_153_64_50',
        # 'pretrained_models/pku-mmdv1/realtime/train_299_64_50',
        'pretrained_models/pku-mmdv1/original/train_9_50_64_50',
        # 'pretrained_models/pku-mmdv1/original/train_21_50_64_50',
        # 'pretrained_models/pku-mmdv1/original/train_69_50_64_50',
        ]

    x = np.arange(0, 51)
    labels = [
        'RT-ST-GCN$_{\Gamma=9}$',
        # 'RT-ST-GCN$_{\Gamma=21}$',
        # 'RT-ST-GCN$_{\Gamma=69}$',
        # 'RT-ST-GCN$_{\Gamma=153}$',
        # 'RT-ST-GCN$_{\Gamma=299}$',
        'ST-GCN$_{L=50,\Gamma=9}$',
        # 'ST-GCN$_{L=50,\Gamma=21}$',
        # 'ST-GCN$_{L=50,\Gamma=69}$'
        ]

    colors = get_color_gradient("#8A5AC2", "#3575D5", len(labels))

    fig = plt.figure()
    gs = fig.add_gridspec(len(paths), hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    for i in range(len(axs)):
        with open('{0}/train-validation-curve.csv'.format(paths[i]),'r') as f:
        # with open('{0}/accuracy-curve.csv'.format(paths[i]),'r') as f:
            y = np.genfromtxt(f, delimiter=',', dtype=np.float32)[1:,1:]

        axs[i].stairs(np.abs(y[::-1,:2].sum(axis=1)-y[::-1,2:].sum(axis=1)),x,fill=True,color=colors[i])
        # axs[i].stairs(np.abs(y[::-1,0]-y[::-1,1]),x,fill=True,color=colors[i])
        axs[i].label_outer()
        axs[i].set_frame_on(False)
        axs[i].set_ylabel(labels[i],fontsize='large',rotation=0)
        # axs[i].set_yticks([0,0.1,0.2])
        axs[i].yaxis.set_label_coords(-.2,.3)
        axs[i].margins(x=0)

    fig.tight_layout()
    plt.show()


plot_accuracy_gap()
# plot_confusion_matrix_rt()
# plot_segmentation_masks()
