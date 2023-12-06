import numpy as np

def image_to_im2col(input_image, block_x, block_y):
    """
    Конвертирует двумерное изображение в формат im2col.
    
    Args:
        input_image (numpy.ndarray): Двумерное изображение.
        block_x (int): Размер блока в направлении X.
        block_y (int): Размер блока в направлении Y.
    
    Returns:
        numpy.ndarray: Результирующий массив в формате im2col.
    """
    shape = (input_image.shape[0] - block_x + 1, input_image.shape[1] - block_y + 1, block_x, block_y)
    strides = (input_image.strides[0], input_image.strides[1], input_image.strides[0], input_image.strides[1])
    patches = np.lib.stride_tricks.as_strided(input_image, shape=shape, strides=strides)
    return patches.reshape(block_x * block_y, -1)

def im2col_to_image(input_im2col, image_x, image_y, kernel_x, kernel_y):
    """
    Конвертирует массив обратно в изображение из формата im2col.
    
    Args:
        input_im2col (numpy.ndarray): Массив в формате im2col.
        image_x (int): Ширина исходного изображения.
        image_y (int): Высота исходного изображения.
        kernel_x (int): Ширина фильтра.
        kernel_y (int): Высота фильтра.
    
    Returns:
        numpy.ndarray: Результирующее изображение.
    """
    kernel_channels = input_im2col.shape[0]
    values_in_row = input_im2col.shape[1] // (image_x - kernel_x + 1)
    result = np.zeros((kernel_channels, input_im2col.shape[1] // values_in_row, values_in_row), dtype=input_im2col.dtype)
    
    for c in range(kernel_channels):
        for i in range(input_im2col.shape[1]):
            result[c][i // values_in_row][i % values_in_row] = input_im2col[c][i]
    return result

def im2col_conv_layer(input_image, kernels):
    """
    Применяет сверточный слой с использованием массива im2col и фильтров.
    
    Args:
        input_image (numpy.ndarray): Двумерное изображение.
        kernels (numpy.ndarray): Фильтры:
            - Первая размерность: количество фильтров.
            - Вторая, третья размерности: размеры фильтров.
    
    Returns:
        numpy.ndarray: Результирующий массив после применения сверточного слоя.
    """
    kernel_channels, kernel_x, kernel_y = kernels.shape
    im2col_image = image_to_im2col(input_image, kernel_x, kernel_y)
    converted_kernels = np.reshape(kernels, (kernel_channels, -1))
    conv_result = np.dot(converted_kernels, im2col_image)
    return im2col_to_image(conv_result, input_image.shape[0], input_image.shape[1], kernel_x, kernel_y)

def reference_conv_layer(input_image, kernel):
    """
    Применяет референсную операцию свертки для тензора с одним фильтром
    
    Args:
        input_image (numpy.ndarray): Двумерное изображение.
        kernel (numpy.ndarray): Фильтр свертки.

    Returns:
        numpy.ndarray: Результирующий массив после применения референсной свертки.
    """
    kernel_x, kernel_y = kernel.shape
    result = np.zeros((input_image.shape[0] - kernel_x + 1, input_image.shape[1] - kernel_y + 1))
    
    for x in range(input_image.shape[0] - kernel_x + 1):
        for y in range(input_image.shape[1] - kernel_y + 1):
            result[x][y] = np.sum(input_image[x:x + kernel_x, y:y + kernel_y] * kernel)
    
    return result

# Test main
if __name__ == "__main__":
    # Пример 1: Простой случай
    input_image_1 = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
    kernels_1 = np.array([[[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]]])

    im2col_result_1 = im2col_conv_layer(input_image_1, kernels_1)
    reference_result_1 = reference_conv_layer(input_image_1, kernels_1[0])

    print("Example 1:")
    print("im2col result:")
    print(im2col_result_1)
    print("Reference result:")
    print(reference_result_1)
    print("\n")

    # Пример 2: Более сложный случай
    input_image_2 = np.array([[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15]])
    kernels_2 = np.array([[[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]],
                          [[0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0]]])

    im2col_result_2 = im2col_conv_layer(input_image_2, kernels_2)
    reference_result_2 = reference_conv_layer(input_image_2, kernels_2[0])

    print("Example 2:")
    print("im2col result:")
    print(im2col_result_2)
    print("Reference result:")
    print(reference_result_2)
    print("\n")

    # Пример 3: Сложный случай
    input_image_3 = np.random.randint(0, 255, size=(128, 128))
    kernels_3 = np.random.rand(5, 5, 3)

    im2col_result_3 = im2col_conv_layer(input_image_3, kernels_3)
    reference_result_3 = reference_conv_layer(input_image_3, kernels_3[0])

    print("Example 3:")
    print("im2col result:")
    print(im2col_result_3)
    print("Reference result:")
    print(reference_result_3)
