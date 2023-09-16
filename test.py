import cv2
import numpy as np

# 비디오 스트림 초기화 (웹캠을 사용하려면 0을 사용하고, 비디오 파일을 사용하려면 파일 경로를 지정하세요.)
cap = cv2.VideoCapture(1)

# 배경 캡쳐 (초기 프레임을 배경으로 사용)
_, background = cap.read()

while True:
    # 프레임 읽기
    _, frame = cap.read()

    # 배경과 현재 프레임 간의 차이 계산
    diff = cv2.absdiff(background, frame)

    # 차이 이미지를 그레이스케일로 변환
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 이진화 (차이가 있는 부분은 흰색, 배경은 검은색)
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

    # 모든 객체의 윤곽선을 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 객체에 대한 윤곽선 그리기
    frame_copy = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 일정 크기 이상의 객체만 고려 (조절 가능)
            cv2.drawContours(frame_copy, [contour], -1, (0, 255, 0), 2)

    # 결과 프레임 보여주기
    cv2.imshow('Object Detection', frame_copy)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 스트림 종료
cap.release()
cv2.destroyAllWindows()
