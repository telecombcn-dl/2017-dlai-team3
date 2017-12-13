from data_input import DataInput
from PIL import Image
import numpy as np
from scipy.misc import imsave
from skimage.exposure import histogram


if __name__ == '__main__':
    threshold = 700
    dataset = DataInput("/storage/dataset", "/storage/dataset_videos/cropped_videos/outputb", "train")
    items_faces, items_audio = dataset.get_items()
    input_images = np.empty([len(items_faces), 64, 64, 3])
    count = 0
    index = [0, 3, 6, 8, 9, 17, 21, 29]
    references = np.empty(shape=[len(index), 64, 10, 3])
    hist_references = np.empty(shape=[len(index), 256])
    bins_references = np.empty(shape=[len(index), 257])
    ind_count = 0
    for ind in index:
        reference = Image.open(items_faces[ind])
        reference = np.asarray(reference, dtype=float)
        reference = reference[:, 0:10, :]
        references[ind_count] = reference
        hist_reference, bins_reference = np.histogram(reference, bins=256, range=(0, 255))
        hist_references[ind_count] = hist_reference
        bins_references[ind_count] = bins_reference
        ind_count += 1

    hist_reference_mean = np.mean(hist_references, axis=0)
    bins_reference_mean = np.mean(bins_references, axis=0)

    distance = 0
    for face in items_faces[0:10]:
        input_image = Image.open(face)
        input_image = np.asarray(input_image, dtype=float)
        input_image_segment = input_image[:, 0:10, :]
        hist_input, bins_input = np.histogram(input_image_segment, bins=256, range=(0, 255))
        distance = np.sum(np.abs(hist_input - hist_reference_mean))
        if distance > threshold:
            input_image = np.flip(input_image, axis=1)
            imsave(str(str(count) + ".jpg"), np.asarray(input_image, dtype=int))
        input_images[count] = input_image
        count += 1
