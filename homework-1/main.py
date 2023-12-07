import cv2
import numpy as np
import pathlib

basePath = pathlib.Path().resolve()

# Загрузка исходного изображения
image = cv2.imread(pathlib.Path().joinpath(basePath, 'lenna.png').as_posix())

# 1. Найти лицо на изображении
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # 2. Отступить на 10% от границ лица и получить этот фрагмент изображения
    padding = int(0.1 * min(w, h))
    face_roi = image[y - padding:y + h + padding, x - padding:x + w + padding]

    # 3. Получить изображение краев
    canny_edges = cv2.Canny(face_roi, 100, 200)

    # 4. Найти угловые точки на первичном изображении и добавить их в изображение границ
    gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_face_roi, 100, 0.01, 10)
    corners = np.intp(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(canny_edges, (x, y), 3, 255, -1)

    # 5. Применить морфологическую операцию наращивания к изображению границ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)

    # 6. Сгладить полученное изображение краев гауссовским фильтром 5 на 5
    smoothed_edges = cv2.GaussianBlur(dilated_edges, (5, 5), 0)

    # 7. Получить нормализованное изображение M
    normalized_image = smoothed_edges / 255.0

    # 8. Сгладить первичное изображение гауссовским фильтром 7 на 7
    smoothed_image = cv2.GaussianBlur(face_roi, (7, 7), 0)

    # 9. Перевести первичное изображение в пространство HSV и увеличить насыщенность цвета
    hsv_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype(float)   # Convert to floating-point

    hsv_image[:, :, 1] *= 1.05

    hsv_image = hsv_image.astype(np.uint8) # Convert back to uint8
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    # 10. Улучшить четкость изображения
    sharpened_image = cv2.filter2D(hsv_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # 11. Финальная фильтрация
    result_image = np.zeros_like(sharpened_image)
    for i in range(3):
        result_image[:, :, i] = normalized_image * sharpened_image[:, :, i] + (1 - normalized_image) * smoothed_image[:, :, i]

    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Face", face_roi)
    cv2.imshow("Canny Edges", canny_edges)
    cv2.imshow("Dilated Edges", dilated_edges)
    cv2.imshow("Smoothed Edges", smoothed_edges)
    cv2.imshow("Normalized Image", normalized_image)
    cv2.imshow("Smoothed Image", smoothed_image)
    cv2.imshow("Sharpened Image", sharpened_image)
    cv2.imshow("Result Image", result_image)

    cv2.waitKey(0)

cv2.destroyAllWindows()