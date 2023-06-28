from keras import utils


def convert_mc2real(mc_image, generator, segmentator):
    labels = segmentator.segment(mc_image[None, ...])
    labels = utils.to_categorical(labels, 25)
    generated = generator(labels)[0]
    return generated
