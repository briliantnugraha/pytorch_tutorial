import os, sys, numpy as np, matplotlib.pyplot as plt
from time import time
from PIL import Image
sys.path.insert(0, './')
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
try:
	print("...Check your packages: pip install apex yacs cython tqdm...")
	# os.system('pip install apex yacs cython tqdm')
except:
	print("[ERROR LOG] There is some error in installing your packages")


config_file = "configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


coco_demo = COCODemo(
    cfg,
    min_image_size=500,
    confidence_threshold=0.7,
)
pil_image = Image.open(sys.argv[1])
image = np.array(pil_image)[:, :, [2, 1, 0]]

start_time = time()
predictions = coco_demo.run_on_opencv_image(image)
print('...runtime: {:.2f}s'.format(time()-start_time))
predictions = predictions[:,:,::-1]

save_img_path = sys.argv[1].split('.')
save_img_name = sys.argv[1]
save_img_path[0] = save_img_path[0] + '_out.'
save_img_path = ''.join(save_img_path)
print('input img: {}, output img: {}'.format(save_img_name, save_img_path))
plt.imsave(save_img_path, predictions)
# plt.imsave(sys.argv[2], predictions)