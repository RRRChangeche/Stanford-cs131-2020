import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image[start_col:start_col+num_cols, start_row:start_row+num_rows,:]
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5*(image**2)
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    factor_row = input_rows/output_rows
    factor_col = input_cols/output_cols
    for col in range(output_cols):
        for row in range(output_rows):
            output_image[row][col] = input_image[int(row*factor_row)][int(col*factor_col)]
    
    # factor_row = output_rows/input_rows
    # factor_col = output_rows/input_cols
    # col = 0
    # row = 0
    # # input=400*400, ouput=100*100
    # for _ in range(input_cols):
    #     for _ in range(input_rows):
    #         output_image[row][col] = input_image[int(row*factor_row)][int(col*factor_col)]
    
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    # rotate point with angle:theta clockwise
    # x' = cos(theta)-sin(theta)
    # y' = sin(theta)+cos(theta)
    # |x'| = R|x|
    # |y'|    |y|
    # R = | cos(theta) -sin(theta)|
    #     | sin(theta)  cos(theta)|
    # p' = Rp
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    point = np.matmul(R, point)
    return point
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)
    output_rows, output_cols, channels = output_image.shape

    ## YOUR CODE HERE
    '''
    p' = R(p-c) + c
    p = R_-1(p'-c) + c
    '''
    center = np.array([int(input_rows/2), int(input_cols/2)])
    for x in range(output_rows):
        for y in range(output_cols):
            pp = np.array([x, y])
            p = rotate2d(pp-center, theta) + center

            # check valid pixel
            if (0 <= p[0] <= output_rows and 0 <= p[1] <= output_cols):
                output_image[x][y] = input_image[int(p[0])][int(p[1])] 
            else: output_image[x][y] = [0, 0, 0]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
