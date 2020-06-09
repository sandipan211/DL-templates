import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, fully_connected
from tensorflow.nn import softmax_cross_entropy_with_logits_v2, softmax, local_response_normalization, relu
from tensorflow.keras.optimizers import SGD
import skimage                                        # for resizing images
from skimage.util import img_as_ubyte
from sklearn.preprocessing import LabelBinarizer      # while displaying test results
import cifar10_utils
from timeit import default_timer as timer
import pickle
import random

import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


class AlexNet:

  def __init__(self, param_list, attributes):
        
    self.dataset = param_list['dataset']
    self.epochs = param_list['epochs']
    self.batch_size = param_list['batch_size']
    self.learning_rate = param_list['learning_rate']
    self.gpu_mode = param_list['gpu_mode']


    self.num_channels = attributes['num_channels']
    self.num_classes = attributes['num_classes']

    self.input_dim = 227
    # define the flow of the network and the utilities to be used when it'll be ran

    # float32 or float64 ??? ===> float32 is max precision
    self.input_data = tf.placeholder(tf.float32, [None, self.input_dim, self.input_dim, self.num_channels], name = 'input_data')
    self.labels = tf.placeholder(tf.int32, [None, self.num_classes], name = 'labels')

    self.logits = self.architecture()
    # later we want the logit tensor to be used during predictions
    # y = x makes the variable y to point to the same object in x, but y = tf.identity(x) creates a new object with the content of x.....should we use another_logit = tf.identity(logits), or simply logits?
    # just self.logits works..but since we want to give a name to logits variable to load it using that name during test time, we use an identity here
    self.model = tf.identity(self.logits, name = 'logits')
    # in CIFAR-10, classes are mutually exclusive, so we use tf.nn.softmax_cross_entropy_with_logits() and then apply tf.reduce_mean() to get out the loss 
    self.cost = tf.reduce_mean(softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.labels), name = 'cost')
    self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)

    # to evaluate the model, we define two more nodes in the graph:

    # find out the total number of correct predictions
    # use another_logit here ???
    self.correct_preds = tf.equal(tf.argmax(self.model, axis = 1), tf.argmax(self.labels, axis = 1), name = 'correct_preds')

    # correct_preds is a vector of True/False. Convert it to float using tf.cast and take the average to get the accuracy find out the current accuracy. For more info, see "https://stackoverflow.com/questions/41708572/tensorflow-questions-regarding-tf-argmax-and-tf-equal"
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32), name='accuracy')


  def architecture(self):

    # to understand padding, see "https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t"

    # conv2d definition at "https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/layers/conv2d"
    conv1 = conv2d(inputs = self.input_data, num_outputs = 96, kernel_size = [11,11], stride = 4, padding = 'VALID', weights_initializer = tf.random_normal_initializer(stddev = 0.01))
    lrn1 = local_response_normalization(input = conv1, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
    pool1 = max_pool2d(inputs = lrn1, kernel_size = [3,3], stride = 2)


    # relu is the default activation_fn
    # authors initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1. 
    conv2 = conv2d(pool1, num_outputs = 256, kernel_size = [5,5], stride = 1, padding = 'SAME', biases_initializer = tf.ones_initializer(), weights_initializer = tf.random_normal_initializer(stddev = 0.01))
    lrn2 = local_response_normalization(input = conv2, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
    pool2 = max_pool2d(inputs = lrn2, kernel_size = [3,3], stride = 2)

    conv3= conv2d(pool2, num_outputs = 384, kernel_size = [3,3], stride = 1, padding = 'SAME', weights_initializer = tf.random_normal_initializer(stddev = 0.01))
    conv4= conv2d(conv3, num_outputs = 384, kernel_size = [3,3], stride = 1, padding = 'SAME', biases_initializer=tf.ones_initializer(), weights_initializer = tf.random_normal_initializer(stddev = 0.01))
    conv5= conv2d(conv4, num_outputs = 256, kernel_size = [3,3], stride = 1, padding = 'SAME', biases_initializer=tf.ones_initializer(), weights_initializer = tf.random_normal_initializer(stddev = 0.01))


    pool3 = max_pool2d(inputs = conv5, kernel_size = [3,3], stride = 2)
    flattened = flatten(pool3)


    fc1 = fully_connected(flattened, num_outputs = 4096, biases_initializer=tf.ones_initializer(), activation_fn = relu, weights_initializer = tf.random_normal_initializer(stddev = 0.01))
    dr1 = tf.nn.dropout(fc1, 0.5)
    fc2 = fully_connected(dr1, num_outputs = 4096, biases_initializer=tf.ones_initializer(), activation_fn = relu, weights_initializer = tf.random_normal_initializer(stddev = 0.01))
    dr2 = tf.nn.dropout(fc2, 0.5)
    final_fc = fully_connected(dr2, num_outputs = self.num_classes, activation_fn = None)

    # final_fc gives the logits on which softmax_cross_entropy_with_logits_v2() would be applied
    return final_fc


  def get_class_names(self, param_list, dataset_loc):

    if param_list['dataset'] == 'cifar10':
      label_info = pickle.load(open(dataset_loc+'/batches.meta', mode = "rb"))
      return label_info['label_names']



  def display_k_preds(self, k, k_imgs, k_labels, random_k_preds, param_list, dataset_loc):
    
    label_names = self.get_class_names(param_list, dataset_loc)
    num_classes = len(label_names)
    
    # using LabelBinarizer() to work efficiently with the labels...for more info see "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html"
    lb = LabelBinarizer()

    # get the label numbers
    lb.fit(range(num_classes))

    # transform binary labels back to multi-class labels
    label_ids = lb.inverse_transform(np.array(k_labels))
    print('label_ids')

    # define plot area - structure: image and probability distribution bar plots - horizontally stacked - so subplot arguments are (k, 2)
    if k > 1:
      fig, axs = plt.subplots(k, 2, figsize=(12,24))
    else:
      fig, axs = plt.subplots(k, 2, figsize=(5, 5))
    margin = 0.05
    ind = np.arange(num_classes)
    width = (1. - 2. * margin) / num_classes  

    # plot for all the k images 
    for img_index, (this_img, this_label, this_pred) in enumerate(zip(k_imgs, label_ids, random_k_preds)):

      label_matched = False

      # check if actual label of this_img matches with predicted label (the one with highest probability)  
      if label_names[this_label] == label_names[np.argmax(this_pred)]:
        label_matched = True

      # plot the result and store the prediction probabilities if required
      all_pred_names = []
      all_pred_probs = []

      # for this image, plot the required graph and also inform the different probabilities for all classes
      for i, pred in enumerate(this_pred):

        tmp_label_name = label_names[i]
        all_pred_probs.append({tmp_label_name: pred})
        all_pred_names.append(tmp_label_name)

      print('Image [{}]==> Ground truth: {}; Prediction: {}; Match found: {}'.format(img_index, label_names[this_label], label_names[np.argmax(this_pred)], label_matched))
      #print('\tDifferent probabilities predicted:- {}\n'.format(all_pred_probs))
      
      if k > 1:
        # test of multiple images
        axs[img_index][0].imshow(this_img)
        axs[img_index][0].set_title(label_names[this_label])
        axs[img_index][0].set_axis_off()
      
        axs[img_index][1].barh(ind + margin, this_pred, width)
        axs[img_index][1].set_yticks(ind + margin)
        axs[img_index][1].set_yticklabels(all_pred_names)

      else:
        # test of single image
        axs[0].imshow(this_img)
        axs[0].set_title(label_names[this_label])
        axs[0].set_axis_off()
      
        axs[1].barh(ind + margin, this_pred, width)
        axs[1].set_yticks(ind + margin)
        axs[1].set_yticklabels(all_pred_names)        
    
    plt.tight_layout()
      


  def testing(self, param_list, attributes, dataset_loc, saved_model_path, k = 10, only_k = False, given_image = None, actual_class = None):

    # make a new tf graph
    loaded_graph = tf.Graph()
    
    if given_image is None:

      # have to test in the batch mode
      with tf.Session(graph = loaded_graph) as sess:

        # load the variables 
        imported_graph = tf.train.import_meta_graph(saved_model_path + '.meta')
        imported_graph.restore(sess, saved_model_path)
        # for t in tf.get_default_graph().get_operations():
        #   print(t)

        # check if the saved variables were correctly loaded
        # for var in tf.get_collection("variables"):
        #   print('Imported variable: ', var)


        # get the imported variables to run a session on
        imported_input = loaded_graph.get_tensor_by_name('input_data:0')
        imported_labels = loaded_graph.get_tensor_by_name('labels:0')
        imported_logits = loaded_graph.get_tensor_by_name('logits:0')
        imported_accuracy = loaded_graph.get_tensor_by_name('accuracy:0')

        if not only_k:

          # have to print test accuracy of entire test set as well as do random k predictions and show
          batch_test_accuracy = 0
          image_batch_size = param_list['batch_size']
          batch_passes = 0
          # preprocess test_batch using the batch_size given in param_list
          for (image_batch, label_batch) in cifar10_utils.load_resized_and_preprocessed_train_or_test_batch(None, image_batch_size, 'test'):

            # get the accuracy from model
            batch_test_accuracy += sess.run(imported_accuracy, feed_dict = {imported_input:image_batch, imported_labels: label_batch})
            batch_passes += 1

          # after testing is complete, return the average accuracy of the test set
          print('Test accuracy: ' + str(batch_test_accuracy / batch_passes))


        # now test on k random test samples
        test_features, test_labels = pickle.load(open('preprocessed_test_set.p', mode = "rb"))

        # using random.samples() to select k samples and their labels - it helps to randomly pick more than one element from a  or sequence without repeating element
        # first zip the imgs and labels - but return as a list as zip() return iterator: see here "https://docs.python.org/3.3/library/functions.html#zip"
        zipped_data = list(zip(np.array(test_features), test_labels))
       
        # now unzip random samples out of it using zip*- and again return a tuple as zip() returns iterator 
        k_imgs, k_labels = tuple(zip(*random.sample(zipped_data , k)))

        # now convert the test images into AlexNet size
        converted_imgs = []
        for image in k_imgs:
          new_img = skimage.transform.resize(image, (self.input_dim, self.input_dim), mode = 'constant')
          new_img = img_as_ubyte(new_img)
          converted_imgs.append(new_img)

        # run predictions on these k images using the trained logits
        random_k_preds = sess.run(softmax(imported_logits), feed_dict = {imported_input: np.array(converted_imgs), imported_labels: k_labels})

        print('k random preds: ')
        print(random_k_preds)
        # display these predicted probabilities along with the images
        self.display_k_preds(k, k_imgs, k_labels, random_k_preds, param_list, dataset_loc)

    else:

      # have to do a prediction only on a given test image
      # first convert the image to AlexNet size
      new_img = skimage.transform.resize(given_image, (self.input_dim, self.input_dim), mode = 'constant')
      new_img = img_as_ubyte(new_img)

      # making the image and label shapes as required in AlexNet  
      new_img = np.expand_dims(new_img, axis = 0) # ===> from (32,32,3) to (1,32,32,3)
      actual_class = np.expand_dims(actual_class, axis = 0)

      with tf.Session(graph = loaded_graph) as sess:

        # load the variables 
        imported_graph = tf.train.import_meta_graph(saved_model_path + '.meta')
        imported_graph.restore(sess, saved_model_path)

        # check if the saved variables were correctly loaded
        # for var in imported_graph.get_collection("variables"):
        #   print('Imported variable: ', var)

        # get the imported variables to run a session on
        imported_input = loaded_graph.get_tensor_by_name('input_data:0')
        imported_labels = loaded_graph.get_tensor_by_name('labels:0')
        imported_logits = loaded_graph.get_tensor_by_name('logits:0')

        pred = sess.run(softmax(imported_logits), feed_dict = {imported_input: new_img, imported_labels: np.array(actual_class)})
        self.display_k_preds(1, new_img, np.array(actual_class), pred, param_list, dataset_loc)


  def batch_training(self, sess, param_list, data_batch):
   
    batch_train_loss_till_now, batch_train_accuracy_till_now = 0, 0
    image_batch_size = param_list['batch_size']
    batch_passes = 0

    # for each of the 5 given training data_batches, preprocess each of them using the batch_size given in param_list, and then train them
    for (image_batch, label_batch) in cifar10_utils.load_resized_and_preprocessed_train_or_test_batch(data_batch, image_batch_size, 'train'):

      # run the optimizer (backprop), and get training loss and accuracy
      train_opt = sess.run(self.optimizer, feed_dict = {self.input_data: image_batch, self.labels: label_batch})
      train_loss, train_accuracy = sess.run([self.cost, self.accuracy], feed_dict = {self.input_data:image_batch, self.labels: label_batch})
      
      batch_train_loss_till_now += train_loss
      batch_train_accuracy_till_now += train_accuracy
      batch_passes += 1

    # after this training_data_batch (one of five) is complete, return the average loss and accuracy of this training_data_batch
    avg_batch_train_loss = batch_train_loss_till_now / batch_passes
    avg_batch_train_accuracy = batch_train_accuracy_till_now / batch_passes 
    print('Batch avg loss: {}, Batch avg acc: {}'.format(avg_batch_train_loss, avg_batch_train_accuracy))
    return avg_batch_train_loss, avg_batch_train_accuracy



  def training_from_ckpt(self, param_list, validation_set, attributes, saved_model_path):

    # a dictonary for storing losses and accuracies
    net_history = {}
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # set params for early stopping
    monitor_max_early_stopping_epochs = param_list['successive_impr']
    last_improvement = 0
    early_stop = False
    best_loss = 1000000

    # get our preprocessed validation data
    validation_features, validation_labels = validation_set

    # make a new tf graph
    loaded_graph = tf.Graph()

    print('Start training from checkpoint.....')
    # start a tf session to run the alexnet graph
    with tf.Session(graph = loaded_graph) as sess:

      # load the variables 
      imported_graph = tf.train.import_meta_graph(saved_model_path + '.meta')
      imported_graph.restore(sess, saved_model_path)

      # check if the saved variables were correctly loaded
      for var in tf.get_collection("variables"):
        print('Imported variable: ', var)

      # get the imported variables to run a session on
      imported_input = loaded_graph.get_tensor_by_name('input_data:0')
      imported_labels = loaded_graph.get_tensor_by_name('labels:0')
      imported_logits = loaded_graph.get_tensor_by_name('logits:0')
      imported_accuracy = loaded_graph.get_tensor_by_name('accuracy:0')

      # have two loops - one for the number of training epochs, and another inside it to feed the training data batch-by-batch
      data_batches = attributes['training_batches_given']

      # epoch loop
      epoch = 0
      while epoch < param_list['epochs'] and early_stop is False:

        # timer() to keep track of training time per epoch
        start = timer()
        this_data_batch_loss, this_data_batch_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0

        for data_batch in range(1, data_batches+1):
          
          # since batches are numbered starting from 1
          print('Batch {} / {}'.format(data_batch, data_batches))
          avg_batch_train_loss, avg_batch_train_accuracy = self.batch_training(sess, param_list, data_batch)
          this_data_batch_loss += avg_batch_train_loss
          this_data_batch_accuracy += avg_batch_train_accuracy
          
        epoch_train_loss = this_data_batch_loss / (attributes['training_samples'] / attributes['per_data_batch_samples'])
        epoch_train_accuracy = this_data_batch_accuracy / (attributes['training_samples'] / attributes['per_data_batch_samples'])


        # after an epoch is over, do a run over the validation set
        for (image_batch, label_batch) in cifar10_utils.extract_image_batch(validation_features, validation_labels, param_list['batch_size']):

          loss, accuracy = sess.run([self.cost, self.accuracy], feed_dict = {input_data:image_batch, labels: label_batch})

          val_loss += loss
          val_accuracy += accuracy

        epoch_val_loss = val_loss / (validation_features.shape[0] / param_list['batch_size'])
        epoch_val_acc = val_accuracy / (validation_features.shape[0] / param_list['batch_size'])
         
        end = timer()

        # check for early stopping
        if epoch_val_loss < best_loss:
          best_loss = epoch_val_loss
          last_improvement = 0
          
        else:
          last_improvement += 1

        if last_improvement > monitor_max_early_stopping_epochs:
          print('No improvement found in validation loss in the last {} epochs....early stopping!'.format(monitor_max_early_stopping_epochs))
          early_stop = True

        # print the epoch results      
        print('Epoch {}/{}, Runtime: {:.1f}s ===> train_loss: {} - train_acc: {} - val_loss: {} - val_acc: {}'
        .format(epoch+1, param_list['epochs'], (end-start), epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_acc))
        
        # store the history of this epoch
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_accuracy)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        epoch += 1

      # store the history of this training run
      net_history['train_loss'] = train_losses
      net_history['train_acc'] = train_accs
      net_history['val_acc'] = val_accs
      net_history['val_loss'] = val_losses

      # save model until now
      saver = tf.train.Saver()
      save_path = saver.save(sess, saved_model_path)

      return net_history


  
  def train(self, param_list, validation_set, attributes, saved_model_path):

    # a dictonary for storing losses and accuracies
    net_history = {}
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # set params for early stopping
    monitor_max_early_stopping_epochs = param_list['successive_impr']
    last_improvement = 0
    early_stop = False
    best_loss = 1000000

    # get our preprocessed validation data
    validation_features, validation_labels = validation_set

    print('Start training.....')
    # start a tf session to run the alexnet graph
    with tf.Session() as sess:

      print('Initializing the variables......')
      sess.run(tf.global_variables_initializer())

      # have two loops - one for the number of training epochs, and another inside it to feed the training data batch-by-batch
      data_batches = attributes['training_batches_given']

      # epoch loop
      epoch = 0
      while epoch < param_list['epochs'] and early_stop is False:

        start = timer()
        this_data_batch_loss, this_data_batch_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0
        val_batches = 0

        for data_batch in range(1, data_batches+1):
          
          print('Batch {} / {}'.format(data_batch, data_batches))
          # since batches are numbered starting from 1
          avg_batch_train_loss, avg_batch_train_accuracy = self.batch_training(sess, param_list, data_batch)
          this_data_batch_loss += avg_batch_train_loss
          this_data_batch_accuracy += avg_batch_train_accuracy
          
        epoch_train_loss = this_data_batch_loss / data_batches
        epoch_train_accuracy = this_data_batch_accuracy / data_batches


        # after an epoch is over, do a run over the validation set
        for (image_batch, label_batch) in cifar10_utils.extract_image_batch(validation_features, validation_labels, param_list['batch_size']):

          loss, accuracy = sess.run([self.cost, self.accuracy], feed_dict = {self.input_data:image_batch, self.labels: label_batch})
          val_batches += 1
          val_loss += loss
          val_accuracy += accuracy

        epoch_val_loss = val_loss / val_batches
        epoch_val_acc = val_accuracy / val_batches
        end = timer()

        if epoch_val_loss < best_loss:
          best_loss = epoch_val_loss
          last_improvement = 0
          
        else:
          last_improvement += 1

        if last_improvement > monitor_max_early_stopping_epochs:
          print('No improvement found in validation loss in the last {} epochs....early stopping!'.format(monitor_max_early_stopping_epochs))
          early_stop = True

        # print the epoch results
        print('Epoch {}/{}, Runtime: {:.1f}s ===> train_loss: {} - train_acc: {} - val_loss: {} - val_acc: {}'
        .format(epoch+1, param_list['epochs'], (end-start), epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_acc))

        # store the history of this epoch
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_accuracy)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        epoch += 1


      # store the history of this training run
      net_history['train_loss'] = train_losses
      net_history['train_acc'] = train_accs
      net_history['val_acc'] = val_accs
      net_history['val_loss'] = val_losses

      # save model until now
      saver = tf.train.Saver()
      save_path = saver.save(sess, saved_model_path)

      return net_history

      
