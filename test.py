import cv2
import time

# 동영상 파일 경로 설정
video_path = 'flybird4.mp4'
cap = cv2.VideoCapture(video_path)

# 배경에서 객체를 분리하기 위해 배경 모델을 사용
fgbg = cv2.createBackgroundSubtractorMOG2()

last_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 배경에서 객체를 분리하기 위해 배경 모델을 사용
    fgmask = fgbg.apply(frame)

    # 이진화를 통해 객체를 감지
    thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)[1]

    # 객체의 윤곽선을 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center_x_sum = 0
    center_y_sum = 0
    num_objects = 0

    for contour in contours:
        if cv2.contourArea(contour) < 100:  # 작은 객체 제외
            continue

        #윤곽선을 둘러싼 최소 사각형
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색 윤곽선 그리기

        # 객체의 중심 좌표 계산
        center_x = x + w // 2
        center_y = y + h // 2

        center_x_sum += center_x
        center_y_sum += center_y
        num_objects += 1

    if num_objects > 0:
        average_center_x = center_x_sum // num_objects
        average_center_y = center_y_sum // num_objects
        print(f"Average Object: ({average_center_x}, {average_center_y})")

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 30ms마다 프레임 업데이트
        break

cap.release()
cv2.destroyAllWindows()
