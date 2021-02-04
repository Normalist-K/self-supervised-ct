import os
import shutil
import random

def split_files(root, rate, folder_name):
    files = {} # {Covid: [Patient ~ Patient (xx)], }
    for name in folder_name:
        f = os.listdir(f'{root}/{name}')
        files[name] = f

    file_num = {} # {Covid: 80, Healthy: 50, }
    for name in folder_name:
        file_num[name] = len(files[name])

    def cmkdir(d):
        if not os.path.isdir(d):
            os.mkdir(d)
            print(f'make directory ...{d[-5:]}')

    dir_test = os.path.join(root, 'test')
    cmkdir(dir_test)
    dir_train = os.path.join(root, 'train')
    cmkdir(dir_train)

    path_origin = {} # {Covid: ~/Covid, }
    path_train = {} # {Covid: ~/train/Covid, }
    path_test = {} # {Covid: ~/test/Covid}

    for name in folder_name:
        path_origin[name] = os.path.join(root, name)
        path_train[name] = os.path.join(dir_train, name)
        path_test[name] = os.path.join(dir_test, name)

    for path in list(path_train.values()) + list(path_test.values()):
        cmkdir(path)

    for name in folder_name:
        rand_idx = random.sample(range(file_num[name]), file_num[name])
        for c, idx in enumerate(rand_idx):
            f = os.path.join(path_origin[name], files[name][idx])
            if c < round(file_num[name] * rate):
                shutil.move(f, path_train[name])
            else:
                shutil.move(f, path_test[name])

    for path in path_origin.values():
        os.rmdir(path)


root = '/home/opticho/source/SimCLR/datasets/dataset2(3)/train'
rate = 0.8
folder_name = os.listdir(root)
# folder_name = ['covid', 'non']

split_files(root, rate, folder_name)