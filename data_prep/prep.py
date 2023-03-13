import argparse
import numpy as np
import os

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='DatasetPreprocess',
        description='Script for preprocessing the PKU-MMDv1/2 dataset')

    parser.add_argument('path', type=str)

    args = parser.parse_args()

    prep_pkummd(args.path)
