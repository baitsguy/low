
import imageio
from imgaug import augmenters as iaa
import glob
import random

images = []
for dir in ['data/v0/obj_train_data/*.jpg', 'data/v2/obj_train_data/*.jpg', 'data/v3/cats/obj_train_data/*.jpg','data/v3/food/*.jpg']:
    files = glob.glob(dir)
    print(dir)
    print(len(files))
    files = random.sample(files,10)
    for filename in files:
        images += [imageio.imread(filename)]

c = 0
for image in images:
    imageio.imwrite("data/augment/" + str(c) + ".jpg", image)
    c += 1
