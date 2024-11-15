import numpy as np
import pyautogui
import cv2
import time
from hand_tracking import HandTracking
from KalmanFilter import KalmanFilter
from cursor_control import CursorControl

pyautogui.FAILSAFE = False

hand_tracking = HandTracking(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cursor_control = CursorControl(screen_width=1920, screen_height=1080)
kalman_filter = KalmanFilter()

cap = cv2.VideoCapture(0)

# ปรับความละเอียดของกล้องให้ตรงกับหน้าจอ
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

screen_width = pyautogui.size().width
screen_height = pyautogui.size().height

is_slicing = False  # สถานะว่ากำลังฟันผลไม้หรือไม่

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = hand_tracking.process_frame(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[hand_tracking.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # คำนวณตำแหน่ง x, y โดยอิงตามความละเอียดของกล้อง
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)

            # ปรับตำแหน่งให้เป็นสัดส่วนกับขนาดหน้าจอและกลับทิศทางการเคลื่อนที่ในแนวนอน
            screen_x = screen_width - int(x * screen_width / frame_width)  # กลับทิศทางในแนวนอน
            screen_y = int(y * screen_height / frame_height)

            # ใช้ Kalman Filter ในการปรับตำแหน่ง
            kalman_filter.predict()
            kalman_filter.update(np.array([screen_x, screen_y], dtype=np.float32))
            filtered_x, filtered_y = kalman_filter.get_state()

            # เคลื่อนที่เคอร์เซอร์
            cursor_control.move_cursor(filtered_x, filtered_y)

            # เริ่มคลิกเมาส์ค้างไว้เมื่อเริ่มการเคลื่อนที่
            if not is_slicing:
                pyautogui.mouseDown()
                is_slicing = True

    else:
        # ปล่อยเมาส์เมื่อไม่มีการตรวจจับมือ
        if is_slicing:
            pyautogui.mouseUp()
            is_slicing = False

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
