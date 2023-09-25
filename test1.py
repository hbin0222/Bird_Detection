import cv2

# 객체 추적기 초기화 (OpenCV 4.x 이상)
tracker = cv2.TrackerMIL_create()

# 배경 제거 모델 초기화
fgbg = cv2.createBackgroundSubtractorMOG2()

def detect_object(frame):
    # 객체 검출을 위한 전처리 (배경 제거)
    fgmask = fgbg.apply(frame)

    # 이진화를 통해 객체를 감지
    thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)[1]

    # 객체의 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []

    for contour in contours:
        # 윤곽선을 둘러싼 사각형 좌표
        x, y, w, h = cv2.boundingRect(contour)

        # 작은 객체 제외
        if cv2.contourArea(contour) < 10:
            continue

        # 검출된 객체의 좌표를 리스트에 추가
        detected_objects.append((x, y, w, h))

        # 객체의 윤곽선 그리기 (예: 초록색으로 윤곽선 그리기)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return detected_objects

# 이전 프레임과 이전 위치를 초기화
prev_frame = None
prev_bbox = None

# 동영상 파일 경로 설정
video_path = 'flybird4.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_bbox = None  # 현재 위치 초기화

    if prev_frame is not None and prev_bbox is not None:
        # 현재 프레임에서 객체 검출
        detected_objects = detect_object(frame)

        if detected_objects:
            # 첫 번째 검출된 객체를 추적 대상으로 설정
            x, y, w, h = detected_objects[0]
            bbox = (x, y, w, h)

            # 객체 추적 업데이트
            success, bbox = tracker.update(frame)

            if success:
                # 추적된 객체 좌표 출력
                x, y, w, h = map(int, bbox)
                center_x = x + w // 2
                center_y = y + h // 2
                print(f"Object Center Coordinates: ({center_x}, {center_y})")

        # 객체 윤곽선 그리기
        for x, y, w, h in detected_objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 현재 프레임을 이전 프레임으로 설정
    prev_frame = frame
    prev_bbox = current_bbox

    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
