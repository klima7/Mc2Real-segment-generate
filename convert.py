import numpy as np
from keras import utils

from generation.gaugan import GauganPredictor
from segmentation.mapping import mapping


segmentator_generator_mapping = {
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


def convert_mc2real(mc_image, generator, segmentator):
    labels = segmentator.segment(mc_image[None, ...])
    labels = _map_labels(labels)
    labels = utils.to_categorical(labels, 25)
    generated = generator(labels)[0]
    return generated


def _map_labels(labels):
    generator_classes = GauganPredictor.CLASSES
    segmentator_classes = list(mapping.keys())
    mapping_array = _create_classes_mapping_array(
        segmentator_classes,
        generator_classes,
        segmentator_generator_mapping
    )
    mapped_labels = mapping_array[labels]
    return mapped_labels
    
    
def _create_classes_mapping_array(in_classes, out_classes, names_mapping):
    num_mapping = []
    for i in range(len(in_classes)):
        in_name = in_classes[i]
        out_name = names_mapping[in_name]
        out_num = out_classes.index(out_name)
        num_mapping.append(out_num)
    return np.array(num_mapping)
