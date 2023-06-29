import cv2
import numpy as np
from keras import utils
from scipy.ndimage import maximum_filter

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
        
        self.segmentator_mask = None
        self.raw_generator_mask = None
        self.filtered_generator_mask = None
        
    def __create_generator(self):
        generator = GauganPredictor(
            'generation/trained_models/generator.h5',
            'generation/trained_models/encoder.h5'
        )
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

    def __call__(self, mc_image):
        labels = self.segmentator.segment(mc_image[None, ...])
        self.segmentator_mask = np.array(labels)
        labels = self.classes_mapping_array[labels]
        self.raw_generator_mask = np.array(labels)
        labels = labels[...]
        labels = self.__filter_labels_with_median_blur(labels)
        self.filtered_generator_mask = np.array(labels)
        labels = utils.to_categorical(labels, 25)
        generated = self.generator(labels)[0]
        return generated

    def get_last_masks(self):
        segmentator_mask = create_readable_mask(self.segmentator_mask[0, ..., 0]) \
            if self.segmentator_mask is not None else None
        raw_generator_mask = self.__create_readable_generator_mask(self.raw_generator_mask[0, ..., 0]) \
            if self.raw_generator_mask is not None else None
        filtered_generator_mask = self.__create_readable_generator_mask(self.filtered_generator_mask[0, ..., 0]) \
            if self.filtered_generator_mask is not None else None
        return [segmentator_mask, raw_generator_mask, filtered_generator_mask]

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
