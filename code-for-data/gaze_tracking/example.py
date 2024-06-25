import cv2
from gaze_tracking import GazeTracking

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
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # 결과 이미지를 창으로 표시
    # cv2.imshow("Processed Image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(gaze.pupil_left_coords())
    # print(gaze.pupil_right_coords())
    return gaze.pupil_left_coords(), gaze.pupil_right_coords()

if __name__ == "__main__":
    # 처리할 이미지 파일 경로
    image_path = '32008_0.png'

    # 이미지 처리 수행
    # print(process_image(image_path))

    eye_coordin_list = process_image(image_path)
    eye_position_pixel = (process_image(image_path)[0][1] + process_image(image_path)[1][1])/2
    print(eye_position_pixel)

