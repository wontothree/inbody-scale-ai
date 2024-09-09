import cv2
import math
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PersonSegmentation:
    def __init__(self, model_path, desired_height=480, desired_width=480, bg_color=(192, 192, 192), mask_color=(255, 255, 255)):
        self.DESIRED_HEIGHT = desired_height
        self.DESIRED_WIDTH = desired_width
        self.BG_COLOR = bg_color
        self.MASK_COLOR = mask_color

        # ImageSegmenter에 사용할 옵션 생성
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    
        # Image segmenter 생성
        self.segmenter = vision.ImageSegmenter.create_from_options(self.options)

    def resize_and_show(self, image):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (self.DESIRED_WIDTH, math.floor(h/(w/self.DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w/(h/self.DESIRED_HEIGHT)), self.DESIRED_HEIGHT))
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def segment_image(self, image_file_name):
        # 세그먼트할 MediaPipe 이미지 파일 생성
        image = mp.Image.create_from_file(image_file_name)

        # 세그먼트된 이미지의 마스크를 가져옴
        segmentation_result = self.segmenter.segment(image)
        category_mask = segmentation_result.category_mask

        # 출력 세그먼트 마스크를 보여주기 위한 단색 이미지 생성
        image_data = image.numpy_view()

        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:, :, :3] = self.MASK_COLOR

        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:, :, :3] = self.BG_COLOR

        condition = np.stack((category_mask.numpy_view(),) * 4, axis=-1)

        output_image = np.where(condition, fg_image, bg_image)

        # print(f'{image_file_name}의 세그먼트 마스크:')
        self.resize_and_show(output_image)

        # 세그먼트된 이미지의 마스크 반환
        return category_mask.numpy_view()

    def find_highest_pixel(self, category_mask):
        """사람이 감지된 가장 높은 픽셀 찾기."""
        person_pixels = np.where(category_mask > 0)
        if len(person_pixels[0]) == 0:
            return None
        highest_pixel = np.min(person_pixels[0])
        return highest_pixel

    def print_highest_pixel(self, image_file_name):
        category_mask = self.segment_image(image_file_name)
        highest_pixel = self.find_highest_pixel(category_mask)
        print(f'-------------------------------- {image_file_name}의 가장 높은 픽셀: {highest_pixel} --------------------------------')

    def process_images(self, image_filenames):
        for image_file_name in image_filenames:
            self.print_highest_pixel(image_file_name)
    
    # real time capture and segmentation
    def capture_and_segment(self):
        # 웹캠 초기화
        webcam = cv2.VideoCapture(0)

        while True:
            # 웹캠에서 새로운 프레임을 가져옴
            ret, frame = webcam.read()

            if not ret:
                print("프레임을 가져올 수 없습니다.")
                break

            print("사용 방법:")
            print("'s' 키를 입력하여 세그멘테이션을 수행하고 가장 높은 픽셀을 출력합니다.")
            print("'q' 키를 입력하여 프로그램을 종료합니다.")

            # 프레임을 보여줌
            cv2.imshow("Webcam Feed", frame)

            # waiting key 's'
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("Segmentation Start")

                # file name being storaging
                image_file_name = 'currentFrame.jpg'
                
                # save current frame to image file with image_file_name
                cv2.imwrite(image_file_name, frame)

                # semantic segmentation
                category_mask = self.segment_image(image_file_name)

                # finding highest pixel for height estimation
                highest_pixel = self.find_highest_pixel(category_mask)
                print(f'-------------------------------- 가장 높은 픽셀: {highest_pixel} --------------------------------')

            # quit using key 'q'
            elif key == ord('q'):
                break

        # close webcam and every window
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    personSegmentation = PersonSegmentation(model_path='/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/heightEstimation/personSegmentation/deeplabv3.tflite')

    # 웹캠에서 이미지 캡처 및 세그멘테이션 수행
    personSegmentation.capture_and_segment()
