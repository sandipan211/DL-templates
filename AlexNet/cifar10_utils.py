from urllib.request import urlretrieve		# to download datasets from the internet
from os.path import isfile, isdir			# to check for files and folder in the system
from tqdm import tqdm						# to show download progress
import numpy as np 							# for numpy utilities
import pickle								# to save checkpoints
import tarfile			# to extract dataset
import platform
import skimage
import skimage.io
import skimage.transform
from skimage.util import img_as_ubyte

# manual tqdm progressBar
class ProgressBar(tqdm):
  
  last_block = 0

	# created as per the argument requirements or urlretrieve: see here "https://docs.python.org/3/library/urllib.request.html"

  def reportHook(self, block_num = 1, block_size = 1, total_size = None):
    self.total = total_size
    self.update((block_num - self.last_block) * block_size)	#updates block-by-block
    self.last_block = block_num


# download and extract dataset into the required folder
def download_data(dataset_loc, data_folder):

  if isfile('cifar-10-python.tar.gz'):
    print('Zipped dataset already exists in the system')

  else:
  	# updating the tqdm manually: see for reference: "https://medium.com/better-programming/python-progress-bars-with-tqdm-by-example-ce98dbbc9697"

		# see params for tqdm at "https://tqdm.github.io/docs/tqdm/"

    with ProgressBar(desc = 'cifar-10-dataset', unit = 'Block', unit_scale = True, miniters = 1) as bar:

      urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar-10-python.tar.gz', bar.reportHook)

	
	# got zipped dataset....now have to extract it
  if isdir(dataset_loc):
    print('Extracted cifar-10-dataset already exists')

  else:
    with tarfile.open('cifar-10-python.tar.gz') as tar:
      tar.extractall(path=data_folder)
      tar.close()
      print("Extracted dataset!")




def get_features_and_labels(dataset_loc, batch_ID, attributes):

  with open(dataset_loc+'/data_batch_'+str(batch_ID), mode="rb") as file:
  	
    version = platform.python_version_tuple()	# obtaining system's python version
    
    if version[0] == '2':		
 		# in python 2.x
      this_batch = pickle.load(file)
    elif version[0] == '3':
			# in python 3.x
      this_batch = pickle.load(file, encoding = 'latin1')
	
	# once we have the dataset encoded in "latin1" format, we have to extract the features and the labels separately to work with, as per the format of the dataset given in "https://www.cs.toronto.edu/~kriz/cifar.html"

	# in each row of the batch file, 3 channels of 32x32 pixel values are given, so first we reshape it channel-wise.....we can keep that shape....but here we reshape it as (width, height, channels) using transpose (from numpy) 
  features = this_batch['data'].reshape((len(this_batch['data']), attributes['num_channels'], attributes['img_width'], attributes['img_height'])).transpose(0, 2, 3, 1)
  labels = this_batch['labels']


  return features, labels


def normalization(imageset):

	# performing a min-max normalization here
  min_px = np.min(imageset)
  max_px = np.max(imageset)

  imageset = (imageset - min_px) / (max_px - min_px)

  return imageset

def mean_centering(imageset):

	# subtracting batch mean from raw pixel values

	# mean is calculated along each channel of this batch of images
	# For a generic ndarray, you could create a tuple to cover all axes except the last one corresponding to the color channel and then use that for the axis param with np.mean - this is to find mean along the channels
  means = np.mean(imageset, axis = tuple(range(imageset.ndim-1)), dtype = 'float64')
	# shape of means: (3, )

	# subtract channel-wise mean from the imageset using np.subtract() - uses the concept of numpy broadcasting inherently
  imageset = np.subtract(imageset, means)

  return imageset 


def one_hot_encode(labels, attributes):

  encoded_array = np.zeros((len(labels), attributes['num_classes']))

  # in the dataset, labels are stored as a list of 10000 numbers in 0-9, for each data_batch - number at index i is the label of image i; so using this:
  for index, label in enumerate(labels):
    encoded_array[index][label] = 1

  return encoded_array
  

def preprocess_and_save_batch_data(features, labels, saved_filename, attributes):

	# normalize the features first
  #features = normalization(features)

  # perform mean centering 
  #features = mean_centering(features)

	# do one-hot encoding of the labels
  labels = one_hot_encode(labels, attributes)

	# dump it all in a pickle file
  pickle.dump((features, labels), open(saved_filename, 'wb'))



# preprocessing of the dataset
def preprocessing(dataset_loc, attributes):

  num_batches = attributes['training_batches_given']
  validation_features = []
  validation_labels = []

	# There are 5 batches of training data - we first extract the features and labels from them
  for batch_ID in range(1, num_batches+1):
		
    features, labels = get_features_and_labels(dataset_loc, batch_ID, attributes)

		# got the features and labels of this batch. Now extract out a validation set out of it....do it for each batch and keep appending into validation_features and validation_labels to get the final validation set of the data.

		# here, we set the validation set to be 10 % of the training data
    validation_index = int(len(features) * (attributes['validation_data_percentage'] / 100))

		# take the first 90% of training data and preprocess it, and keep saving it into separate files
    saved_filename = "preprocessed_training_batch_" + str(batch_ID) + ".p"
    preprocess_and_save_batch_data(features[:-validation_index], labels[:-validation_index], saved_filename, attributes)

		# preprocessing of the validation data will be done at once on the final validation set obtained

		# better to use extend() instead of append() here because we can keep on iterating the features and labels and keep adding them in chunks
    validation_features.extend(features[-validation_index:])
    validation_labels.extend(labels[-validation_index:])

  saved_filename = "preprocessed_validation_set.p"
	
	# preprocess_and_save_batch_data() needs np arrays as arguments
  preprocess_and_save_batch_data(np.array(validation_features), np.array(validation_labels), saved_filename, attributes)


	# now preprocess the test set - only one batch
  with open(dataset_loc+'/test_batch', mode = "rb") as file:

    version = platform.python_version_tuple()	# obtaining system's python version
    if version[0] == '2':		
			# in python 2.x
      testdata = pickle.load(file)
    elif version[0] == '3':
			# in python 3.x
      testdata = pickle.load(file, encoding = 'latin1')

  test_features = testdata['data'].reshape((len(testdata['data']), attributes['num_channels'], attributes['img_width'], attributes['img_height'])).transpose(0, 2, 3, 1)

  test_labels = testdata['labels']

  saved_filename = "preprocessed_test_set.p"

	# check if np.array() is required
  preprocess_and_save_batch_data(np.array(test_features), np.array(test_labels), saved_filename, attributes)


# upon entering the training module later, we would need both preprocessed training and validation sets
def load_resized_and_preprocessed_validation_set():

  validation_features, validation_labels = pickle.load(open('preprocessed_validation_set.p', mode = "rb"))

  converted = []

	# now convert the downloaded images to match the size used in AlexNet: (227 x 227)
  for feature_set in validation_features:
		# using skimage to resize image: see "https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize" for details
    new_img = skimage.transform.resize(feature_set, (227, 227), mode = 'constant')
    new_img = img_as_ubyte(new_img)
    converted.append(new_img)

  return (np.array(converted), validation_labels)


def extract_image_batch(features, labels, image_batch_size):

  for starting_image in range(0, len(features), image_batch_size):

    # the last batch may contain less than image_batch_size number of images - so find ut the ending_image positions first
    ending_image = min(starting_image + image_batch_size, len(features))

	  # yield is memorry efficient than 'return'. For more info, see "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do?page=1&tab=votes#tab-top"
    yield features[starting_image:ending_image], labels[starting_image:ending_image]

	

def load_resized_and_preprocessed_train_or_test_batch(data_batch, image_batch_size, train_or_test):

  if train_or_test == 'train':
    features, labels = pickle.load(open('preprocessed_training_batch_'+str(data_batch)+'.p', mode = "rb"))
  else:
    features, labels = pickle.load(open('preprocessed_test_set.p', mode = "rb"))

  converted = []

  # convert the images to match the size used in AlexNet: (227 x 227)
  for feature_set in features:
    new_img = skimage.transform.resize(feature_set, (227, 227), mode = 'constant')
    new_img = img_as_ubyte(new_img)
    converted.append(new_img)

  # take out a batch of images as per image_batch_size and return the converted batch of images and corresponding labels 
  return extract_image_batch(np.array(converted), labels, image_batch_size)
  
