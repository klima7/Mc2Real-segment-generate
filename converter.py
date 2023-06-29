from typing import Any
import numpy as np
from keras import utils

from generation.gaugan import GauganPredictor
from segmentation.unet import UnetModel
from segmentation.mapping import mapping


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

    
    def __call__(self, mc_image) -> Any:
        labels = self.segmentator.segment(mc_image[None, ...])
        labels = self.classes_mapping_array[labels]
        labels = utils.to_categorical(labels, 25)
        generated = self.generator(labels)[0]
        return generated
