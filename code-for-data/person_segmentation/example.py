from person_segmentation import PersonSegmentation
import cv2


if __name__ == "__main__":
    # 모델 경로로 PersonSegmentation 클래스 초기화
    segmenter = PersonSegmentation(model_path='deeplabv3.tflite')

    # 이미지 파일 이름
    IMAGE_FILENAMES = ['./test_image/man.jpeg']

    # 이미지 파일을 메모리로 읽어오기
    image = cv2.imread(IMAGE_FILENAMES[0])
    if image is None:
        print(f"Error: Could not read image {IMAGE_FILENAMES[0]}.")
    else:
        # 메모리에서 이미지 처리 및 표시
        category_mask = segmenter.segment_image_from_memory(image)
        highest_pixel = segmenter.find_highest_pixel(category_mask)
        print(f'-------------------------------- {IMAGE_FILENAMES[0]}의 가장 높은 픽셀: {highest_pixel} --------------------------------')