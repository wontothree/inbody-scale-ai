from person_segmentation import PersonSegmentation
import cv2

def capture_and_segment(segmenter):
    # 웹캠 초기화
    webcam = cv2.VideoCapture(0)

    while True:
        # 웹캠에서 새로운 프레임을 가져옴
        ret, frame = webcam.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # 프레임을 보여줌
        cv2.imshow("Webcam Feed", frame)

        # 's' 키 입력을 기다림
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("Segmentation 시작")
            # 세그멘테이션할 이미지 파일 이름
            image_file_name = 'current_frame.jpg'
            # 현재 프레임을 이미지 파일로 저장
            cv2.imwrite(image_file_name, frame)
            # 세그멘테이션 수행
            category_mask = segmenter.segment_image(image_file_name)
            highest_pixel = segmenter.find_highest_pixel(category_mask)
            print(f'-------------------------------- 가장 높은 픽셀: {highest_pixel} --------------------------------')

        # 'q' 키가 눌리면 종료
        elif key == ord('q'):
            break

    # 웹캠 및 모든 창 닫기
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 모델 경로로 PersonSegmentation 클래스 초기화
    segmenter = PersonSegmentation(model_path='deeplabv3.tflite')

    print("사용 방법:")
    print("'s' 키를 입력하여 세그멘테이션을 수행하고 가장 높은 픽셀을 출력합니다.")
    print("'q' 키를 입력하여 프로그램을 종료합니다.")

    # 웹캠에서 이미지 캡처 및 세그멘테이션 수행
    capture_and_segment(segmenter)
