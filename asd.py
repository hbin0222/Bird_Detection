import cv2

# 웹캠 또는 비디오 파일에서 프레임 읽기
cap = cv2.VideoCapture(1)  # 웹캠을 사용하려면 0을 사용하거나 비디오 파일 경로를 지정하세요.

# 초기 배경 생성
_, first_frame = cap.read()
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (25, 25), 0)
    
    # 차이 계산
    frame_delta = cv2.absdiff(first_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]

    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 2000:  # 작은 객체 제외
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 화면에 프레임 표시
    cv2.imshow("Security Feed", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 정리
cap.release()
cv2.destroyAllWindows()
