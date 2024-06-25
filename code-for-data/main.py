import cv2
import os
from gaze_tracking.gaze_tracking import GazeTracking
from person_segmentation.person_segmentation import PersonSegmentation
import numpy as np

def process_images_in_directory(directory, segmenter):
    # 디렉토리 내의 모든 이미지 파일을 처리
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 지원하는 이미지 파일 형식
            image_path = os.path.join(directory, filename)
            highest_pixel = process_image_segmentation(image_path, segmenter)
            if highest_pixel is not None:
                print(f'Highest Pixel in {filename}: {highest_pixel}')
            else:
                print(f'No person detected in {filename}')

def process_image_segmentation(image_path, segmenter):
    # 이미지를 읽어오기
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image {image_path}.")
        return None

    # 세그멘테이션 수행
    category_mask = segmenter.segment_image(image_path)
    highest_pixel = segmenter.find_highest_pixel(category_mask)

    return highest_pixel



def process_image(image_path):
    gaze = GazeTracking()

    # 이미지 파일을 읽어오기
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}.")
        return

    # 이미지를 GazeTracking에 전달하여 분석
    gaze.refresh(frame)

    frame = gaze.annotated_frame()


    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    return gaze.pupil_left_coords(), gaze.pupil_right_coords()



if __name__ == "__main__":
    # 처리할 이미지 파일 경로
    image_path = '32008_0.png'
    segmenter = PersonSegmentation(model_path='./person_segmentation/deeplabv3.tflite')

    eye_coordin_list = process_image(image_path)
    eye_position_pixel = (process_image(image_path)[0][1] + process_image(image_path)[1][1])/2

    height_position_pixel = process_image_segmentation(image_path, segmenter)

    # print("눈 위치 픽셀")
    # print(eye_position_pixel)

    # print("키 위치 픽셀")
    # print(height_position_pixel)

    H, W = 640,480
    DISTANCE = 400 #mm
    VFOV = 55*np.pi/180 
    mm_per_pixel = DISTANCE*np.tan(VFOV/2)/(H/2)
    cm_per_pixel = mm_per_pixel / 10

    eye_position = 168

    pixel_difference = -(height_position_pixel - eye_position_pixel)
    height_estimation = eye_position + cm_per_pixel * pixel_difference

    print(eye_position + cm_per_pixel*pixel_difference)




