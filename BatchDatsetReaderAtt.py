"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
from sklearn.utils import shuffle


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.image_files  = [filename['image'] for filename in self.files]
        self.annotation_files =  [filename['annotation'] for filename in self.files]
        print (len(self.image_files))
        print (len(self.annotation_files))

    def _transform(self, filename, mode):
        image = misc.imread(filename, mode = 'RGB').astype(np.uint8)
        if mode == "images":  # make sure images are of shape(h,w,3)
            image = image[:,:128,:]
        else:
            image = image[:,128:,:]
          #  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='bicubic')
        else:
            resize_image = image
        
        resize_image = resize_image/127.5 - 1.0
        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        epoch_bool = False 
        if self.batch_offset > len(self.image_files):
            # Finished epoch
            epoch_bool = True
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            self.image_files, self.annotation_files = shuffle(self.image_files, self.annotation_files)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset

        current_image_batch = self.image_files[start:end]
        current_annotation_batch = self.annotation_files[start:end]

        list1 = []
        list2 = []
     #   try:
        for filename in current_image_batch:
                list1.append(self._transform(filename,'images'))
                list2.append(self._transform(filename,'annotations'))
         #   image_batch = np.array([self._transform(filename,'images') for filename in current_image_batch])
          #  annotation_batch = np.array([self._transform(filename,'annotations') for filename in current_annotation_batch])
       # except:
       #     print(filename + "_error")

        image_batch = np.array(list1)
        annotation_batch = np.array(list2)
      #  print([current_image_batch])
      #  print([filename for filename in current_annotation_batch])
        return image_batch , annotation_batch, epoch_bool


    def get_random_batch(self, batch_size):
        list1 = []
        list2 = []
        indexes = np.random.randint(0, len(self.image_files), size=[batch_size]).tolist()
        for i in indexes:
            filename = self.image_files[i]
            list1.append(self._transform(filename,'images'))
            list2.append(self._transform(filename,'annotations'))

        image_batch = np.array(list1)
        annotation_batch = np.array(list2)
        return image_batch, annotation_batch
