import os
import sys
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

from sklearn.decomposition import PCA

import tensorflow as tf



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    
    print("\n\t****CALCULATING ALL LAYERS/TENSORS*****\n\n")
    for op in self.graph.get_operations():
      print (op.name,' | VALUES | : ',op.values())
      if op.name == 'decoder/decoder_conv0_pointwise/Conv2D':
        x = op.values()

    print("\nSpecific tensor found : ",x)
    
    inter_layer_batch_seg_map = self.sess.run(
        'decoder/decoder_conv1_pointwise/Conv2D:0',
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    inter_seg_map = inter_layer_batch_seg_map[0]

    print('Inter Layer seg_map shape : ',inter_layer_batch_seg_map.shape)

    N = inter_layer_batch_seg_map.shape[1]*inter_layer_batch_seg_map.shape[2]
    C = inter_layer_batch_seg_map.shape[-1]
    
    print("N = ", N)
    print("C = ", C)

    X = np.reshape(inter_layer_batch_seg_map, [N, C])

    print('Data shape before PCA : {}'.format(inter_layer_batch_seg_map.shape))
    print('\nPerforming PCA .... \n')
    Xreduced = PCA(n_components=3).fit_transform(X)
    print('Data shape after PCA : {}'.format(Xreduced.shape))


    deepfeats_reduced = np.reshape(Xreduced, [inter_layer_batch_seg_map.shape[1], inter_layer_batch_seg_map.shape[2], 3])
    print('Reduced shape : ',deepfeats_reduced.shape)
    plt.imshow(deepfeats_reduced)
    plt.show()

    print("\n\n\n... ENTIRE GRAPH IS SAVED ...\n")
    writer = tf.summary.FileWriter('/tmp/tftut/1')
    writer.add_graph(self.sess.graph)
    
    return resized_image, seg_map




def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  print("\n... VISUALIZING IMAGE ...\n")
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


MODEL_NAME = 'xception_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

MODEL = DeepLabModel("/home/vaggelisbarb/models/research/deeplab/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz")
print('model loaded successfully!')


IMAGE_DIR = '/home/vaggelisbarb/crashcourse-tensorflow'

def run_image(image_name):
    try:
        image_path = os.path.join(IMAGE_DIR, image_name)
        orignal_im = Image.open(image_path)
    except IOError:
        print('Failed to read image from %s.' % image_path)
        return 
    print('Running Deeplab on image : %s ...' % image_name)
    resized_im, seg_map = MODEL.run(orignal_im)
    
    vis_segmentation(resized_im, seg_map)

run_image(str(sys.argv[1]))


