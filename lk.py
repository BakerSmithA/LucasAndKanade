import numpy as np
import cv2
from collections import namedtuple


# Stores spatial and temporal gradients.
Gradient = namedtuple('Gradient', 'ix iy it')


def gradient_x(block: np.array) -> float:
    """
    :param block: a 2x2 block of an image.
    :return: the spatial gradient in the x direction.
    """
    return (block[:,1] - block[:,0]).sum()


def gradient_y(block: np.array) -> float:
    """
    :param block: a 2x2 block of an image.
    :return: the spatial gradient in the y direction.
    """
    return (block[0,:] - block[1,:]).sum()


def gradient_t(block1: np.array, block2: np.array) -> float:
    """
    :param block1: a 2x2 block of image in frame t.
    :param block2: a 2x2 block of the corresponding section of image in frame t+1.
    :return: the spatial gradient in the time direction.
    """
    return (block2 - block1).sum()


def estimate_gradients(block1: np.array, block2: np.array) -> Gradient:
    """
    :param block1: a 2x2 block of image in frame t.
    :param block2: a 2x2 block of the corresponding section of image in frame t+1.
    :return: the spatial and temporal gradients for the block.
    """

    # Normalise by dividing by 4
    ix = (gradient_x(block1) + gradient_x(block2)) / 4
    iy = (gradient_y(block1) + gradient_y(block2)) / 4
    it = gradient_t(block1, block2) / 4

    return Gradient(ix, iy, it)


if __name__ == '__main__':
    # Web camera
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()