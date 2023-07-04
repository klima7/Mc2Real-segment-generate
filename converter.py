import random
from glob import glob
import cv2
import numpy as np
from keras import utils
from scipy.ndimage import maximum_filter
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import MeanSquaredError

from generation.gaugan import GauganPredictor
from segmentation.unet import UnetModel
from segmentation.mapping import mapping, create_readable_mask


class Mc2RealConverter:
    
    SEGMENTATOR_GENERATOR_MAPPING = {
        'sky': 'sky',
        'water': 'water',
        'sand': 'sand',
        'tree': 'tree',
        'plant': 'plant',
        'grass': 'grass',
        'flower': 'flower',
        'stone': 'mountain',
        'dirt': 'mountain',
        'snow': 'grass',
        'unknown': 'unknown',
    }
    
    def __init__(self):
        self.generator = self.__create_generator()
        self.segmentator = self.__create_segmentator()
        self.classes_mapping_array = self.__create_classes_mapping_array()
        self.generator_colors = self.__generate_colors(len(GauganPredictor.CLASSES))
        
        self._segmentator_mask = None
        self._raw_generator_mask = None
        self._filtered_generator_mask = None
        self._img_without_opt = None
        self._img_with_opt = None
        
    def __create_generator(self):
        generator = GauganPredictor(
            'generation/trained_models/generator.h5',
            'generation/trained_models/encoder.h5'
        )
        generator.gen.trainable = False
        return generator
    
    def __create_segmentator(self):
        segmentator = UnetModel()
        segmentator.load_weights('segmentation/trained_models/unet.h5')
        return segmentator
    
    def __create_classes_mapping_array(self):
        in_classes = GauganPredictor.CLASSES
        out_classes = list(mapping.keys())
        
        mapping_array = []
        for i in range(len(out_classes)):
            in_name = out_classes[i]
            out_name = self.SEGMENTATOR_GENERATOR_MAPPING[in_name]
            out_num = in_classes.index(out_name)
            mapping_array.append(out_num)
        return np.array(mapping_array)
    
    @staticmethod
    def __generate_colors(num_of_colors, seed=0):
        colors = np.random.Generator(np.random.PCG64(seed)).integers(32, 255, size=(num_of_colors, 3))
        return colors.astype(np.uint8)
    
    @property
    def last_segmentator_mask(self):
        return create_readable_mask(self._segmentator_mask[0, ..., 0]) \
            if self._segmentator_mask is not None else None

    @property
    def last_raw_generator_mask(self):
        return self.__create_readable_generator_mask(self._raw_generator_mask[0, ..., 0]) \
            if self._raw_generator_mask is not None else None

    @property
    def last_filtered_generator_mask(self):
        return self.__create_readable_generator_mask(self._filtered_generator_mask[0, ..., 0]) \
            if self._filtered_generator_mask is not None else None

    @property
    def last_img_without_opt(self):
        return self._img_without_opt
    
    @property
    def last_img_with_opt(self):
        return self._img_with_opt

    def __call__(self, mc_image, opt_steps=0, seed=None, progress_callback=None):
        labels = self.segmentator.segment(mc_image[None, ...])
        self._segmentator_mask = np.array(labels)

        labels = self.classes_mapping_array[labels]
        self._raw_generator_mask = np.array(labels)

        labels = self.__filter_labels_with_median_blur(labels)
        self._filtered_generator_mask = np.array(labels)

        labels = utils.to_categorical(labels, 25)
        noise = self.get_optimized_noise(labels, mc_image, opt_steps=opt_steps, seed=seed, progress_callback=progress_callback)
        generated = self.generator(labels, noise)[0]
        self._img_with_opt = generated if opt_steps > 0 else None
        return generated
    
    def get_optimized_noise(self, labels, mc_image, opt_steps=100, seed=None,
                            loss_ratio=0.5, progress_callback=None):
        if seed is not None:
            tf.random.set_seed(seed)

        initial_noise = tf.random.normal(shape=(1, 256))

        self._img_without_opt = self.generator(labels, initial_noise)[0]

        noise = tf.Variable(
            initial_value=initial_noise,
            trainable=True,
        )

        mc_image = mc_image*2-1
        mc_image = mc_image[np.newaxis, ...]
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.015)
        
        iterator = tqdm(range(opt_steps))
        for step in iterator:
            with tf.GradientTape() as tape:
                prediction = self.generator.gen([noise, labels])
                l1_loss = tf.math.reduce_sum(tf.math.abs(prediction - mc_image))
                vgg_loss = self.__vgg_loss(prediction, self.__get_random_real_image())
                loss_value = loss_ratio * l1_loss + (1-loss_ratio) * vgg_loss
                
            gradients = tape.gradient(loss_value, [noise])
            optimizer.apply_gradients(zip(gradients, [noise]))
            
            iterator.set_postfix_str(f'Loss: {loss_value.numpy()}')
            
            if progress_callback:
                progress_callback((step+1) / opt_steps)
            
        return noise.numpy()
    
    def __get_random_real_image(self):
        paths = list(glob('dataset_real/*.png'))
        path = random.choice(paths)
        image = cv2.imread(str(path))[None, ..., ::-1]
        return image
    
    @staticmethod
    def __vgg_loss(y_true, y_pred):
        vgg = VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        loss_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv3').output)
        true_features = loss_model(y_true)
        pred_features = loss_model(y_pred)
        mse_loss = MeanSquaredError()(true_features, pred_features)
        return mse_loss

    def __create_readable_generator_mask(self, mask):
        readable_mask = self.generator_colors[mask]
        return readable_mask

    def __filter_labels_with_median_blur(self, labels):
        fmaps = cv2.medianBlur(labels[0].astype(np.uint8), 27)[..., np.newaxis]
        unique, counts = np.unique(fmaps, return_counts=True)
        for u, c in zip(unique, counts):
            if c < 1000:
                fmaps[fmaps == u] = 0
        fmaps = maximum_filter(fmaps, 5)
        return fmaps[np.newaxis, ...]
