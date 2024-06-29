import random
import os

wd = os.path.dirname(__file__)

dataset_path = os.path.join(wd, 'archive/train')
output_path = os.path.join(wd, 'dArchive')

os.mkdir(output_path)
os.mkdir(os.path.join(output_path, 'train'))
os.mkdir(os.path.join(output_path, 'val'))
os.mkdir(os.path.join(output_path, 'test'))

train_ratio = 0.7
val_ratio = 0.2

classes = os.listdir(dataset_path)

for i in range(30):

    os.mkdir(os.path.join(output_path, 'train', classes[i]))
    os.mkdir(os.path.join(output_path, 'val', classes[i]))
    os.mkdir(os.path.join(output_path, 'test', classes[i]))

    class_path = os.path.join(dataset_path, classes[i])
    images = os.listdir(class_path)

    train_value = int(train_ratio * len(images))
    val_value = int(val_ratio * len(images))

    random.shuffle(images)

    for idx, img in enumerate(images):
        img_path = os.path.join(class_path, img)

        if idx < train_value:
            os.system(f'cp {img_path} {os.path.join(output_path, "train", classes[i], img)}')
        elif idx < train_value + val_value:
            os.system(f'cp {img_path} {os.path.join(output_path, "val", classes[i], img)}')
        else:
            os.system(f'cp {img_path} {os.path.join(output_path, "test", classes[i], img)}')

print("Operation Compelet")