import argparse
import numpy as np
import os
import random


def permute(dir):
    for f in os.listdir(dir):
        data = np.transpose(np.float32(np.load('{0}/{1}'.format(dir, f))), (3,0,2,1))
        with open('{0}/{1}'.format(dir, f), 'wb') as file:
            np.save(file, np.ascontiguousarray(data))


def prep_pkummd(dir):
    with open('{0}/cross-view.txt'.format(dir)) as f_xview:
        xview_lines = f_xview.readlines()
        train_xview = xview_lines[1].split(', ')

    # with open('{0}/cross-subject.txt'.format(dir)) as f_xsub:
    #     xsub_lines = f_xsub.readlines()
    #     train_xsub = xsub_lines[1].split(', ')

    for f_in in os.listdir('{0}/features'.format(dir)):
        features = np.loadtxt('{0}/features/{1}'.format(dir, f_in), dtype=np.float32)
        features = np.ascontiguousarray(np.transpose(features.reshape(features.shape[0], 2, 25, 3), (3,0,2,1)))
        
        labels = np.loadtxt('{0}/labels/{1}'.format(dir, f_in),delimiter=",",dtype=np.int32)
        d = np.zeros(features.shape[0],dtype=np.int32)
        for row in labels:
            d[row[1]:row[2]] = row[0]

        p_xview = 'train' if f_in.split('.')[0] in train_xview else 'val'
        # p_xsub = 'train' if f_in.split('.')[0] in train_xsub else 'val'

        with open('{0}/{1}/features/{2}.npy'.format(dir,p_xview,f_in.split('.')[0]), 'wb') as f_out:
            np.save(f_out, features)

        with open('{0}/{1}/labels/{2}.csv'.format(dir,p_xview,f_in.split('.')[0]), 'w') as f_out:
            np.savetxt(f_out,d,delimiter=',')

        # with open('{0}/xsub/{1}/features/{2}.npy'.format(dir,p_xsub,f_in.split('.')[0]), 'wb') as f_out:
        #     np.save(f_out, features)

        # with open('{0}/xsub/{1}/labels/{2}.csv'.format(dir,p_xsub,f_in.split('.')[0]), 'w') as f_out:
        #     np.savetxt(f_out,d,delimiter=',')

        os.remove('{0}/features/{1}'.format(dir, f_in))
        os.remove('{0}/labels/{1}'.format(dir, f_in))


def prep_imu_fogit_ABCD(dir):
    files = []

    for d in os.listdir('{0}/annotation'.format(dir)):
        for f in os.listdir('{0}/annotation/{1}'.format(dir, d)):
            file = ''.join(f.split('_alltypes')).split('.')[0]
            files.append(file)
            # update data dimension order to (C,L,V)
            data = np.load('{0}/imu/{1}/{2}.npy'.format(dir, d, file)).astype(np.float32)
            data = np.ascontiguousarray(np.transpose(data.reshape((*data.shape,1)), (1,2,0,3)))
            np.save('{0}/imu/{1}/{2}.npy'.format(dir, d, file), data)
            # turn label numpy files into .csv and rename to match data files
            with open('{0}/annotation/{1}/{2}.csv'.format(dir, d, file),'w') as f_labels:
                labels = np.load('{0}/annotation/{1}/{2}'.format(dir, d, f)).astype(np.int32)
                np.savetxt(f_labels, np.ascontiguousarray(np.transpose(labels,(1,0))), delimiter=',')
    
    random.shuffle(files)
    file_set = set(files)
    val_set = set(random.sample(file_set, int(len(file_set)*0.3)))
    train_set = file_set - val_set

    os.makedirs('{0}/train/features'.format(dir))
    os.makedirs('{0}/train/labels'.format(dir))
    os.makedirs('{0}/val/features'.format(dir))
    os.makedirs('{0}/val/labels'.format(dir))

    for f in train_set:
        os.system('mv {0}/annotation/{1}/{2}.csv {0}/train/labels'.format(dir, f.split('_')[0], f))
        os.system('mv {0}/imu/{1}/{2}.npy {0}/train/features'.format(dir, f.split('_')[0], f))

    for f in val_set:
        os.system('mv {0}/annotation/{1}/{2}.csv {0}/val/labels'.format(dir, f.split('_')[0], f))
        os.system('mv {0}/imu/{1}/{2}.npy {0}/val/features'.format(dir, f.split('_')[0], f))

    os.system('rm -r {0}/annotation'.format(dir))
    os.system('rm -r {0}/imu'.format(dir))

    with open('{0}/split.txt'.format(dir), 'w') as f:
        f.write('train:'+', '.join(list(train_set))+'\nval:'+', '.join(list(val_set)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='DatasetPreprocess',
        description='Script for preprocessing the PKU-MMDv1/2 dataset')

    parser.add_argument('path', type=str)

    args = parser.parse_args()

    prep_imu_fogit_ABCD(args.path)
