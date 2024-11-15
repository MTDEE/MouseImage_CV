import numpy as np
import pyautogui
import cv2
import time
from hand_tracking import HandTracking
from KalmanFilter import KalmanFilter
from cursor_control import CursorControl

# ปิดฟีเจอร์ fail-safe ของ pyautogui
pyautogui.FAILSAFE = False

# กำหนดค่าเริ่มต้น
hand_tracking = HandTracking(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
cursor_control = CursorControl(screen_width=1920, screen_height=1080)
kalman_filter = KalmanFilter()  # สร้างอ็อบเจ็กต์ KalmanFilter

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# กำหนดความเร็วในการขยับเมาส์ (ค่าตั้งต้นเป็น 2.0)
speed_factor = 1.0  # ปรับค่าได้ตามต้องการ

# สถานะของการคลิก
last_click_time = 0
click_delay = 0.1  # ระยะเวลาในการคลิกซ้ำ

# เก็บข้อมูลตำแหน่ง Y ของนิ้วกลางจากเฟรมก่อนหน้า
last_middle_finger_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับมือ
    results = hand_tracking.process_frame(frame)

    # ตรวจจับตำแหน่งของนิ้วโป้ง, นิ้วชี้ และนิ้วกลาง
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # ดึงตำแหน่งของปลายนิ้วโป้ง, นิ้วชี้ และนิ้วกลาง
            index_finger_tip = hand_landmarks.landmark[hand_tracking.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[hand_tracking.mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmarks.landmark[hand_tracking.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # คำนวณตำแหน่ง x, y ของเคอเซอร์ (แปลงจากค่าระดับ 0-1 เป็นค่าพิกเซล)
            x = int(index_finger_tip.x * 1920 * speed_factor)  # คูณกับ speed_factor เพื่อเพิ่มความเร็ว
            y = int(index_finger_tip.y * 1080 * speed_factor)  # คูณกับ speed_factor เพื่อเพิ่มความเร็ว

            # จำกัดค่าตำแหน่งให้ไม่เกินขอบเขตหน้าจอ
            x = max(0, min(1920, x))  # จำกัดค่าของ x
            y = max(0, min(1080, y))  # จำกัดค่าของ y

            # ทำนายตำแหน่งก่อนหน้าโดยใช้ Kalman Filter
            kalman_filter.predict()

            # อัปเดต Kalman Filter ด้วยตำแหน่งปัจจุบัน
            kalman_filter.update(np.array([x, y], dtype=np.float32))

            # รับค่าผลลัพธ์ที่กรองจาก Kalman Filter
            filtered_x, filtered_y = kalman_filter.get_state()

            # คำนวณการขยับเคอร์เซอร์ด้วยการคูณด้วย speed_factor
            cursor_control.move_cursor(filtered_x, filtered_y)

            # ตรวจสอบว่าปลายนิ้วโป้งแตะนิ้วชี้ (คลิกซ้าย)
            thumb_index_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
            if thumb_index_distance < 0.05:
                current_time = time.time()
                if current_time - last_click_time > click_delay:
                    cursor_control.click()  # คลิกซ้าย
                    last_click_time = current_time

            # ตรวจสอบว่าปลายนิ้วโป้งชิดกับนิ้วกลาง (คลิกขวา)
            thumb_middle_distance = ((thumb_tip.x - middle_finger_tip.x) ** 2 + (thumb_tip.y - middle_finger_tip.y) ** 2) ** 0.5
            if thumb_middle_distance < 0.05:
                cursor_control.right_click()  # คลิกขวา

            # ในส่วนของการตรวจจับนิ้ว
            index_middle_distance = ((index_finger_tip.x - middle_finger_tip.x) ** 2 +
                                     (index_finger_tip.y - middle_finger_tip.y) ** 2) ** 0.5

            if index_middle_distance < 0.06:  # ปรับค่าขีดจำกัดนี้ตามความเหมาะสม
                # คำนวณการเปลี่ยนแปลงในตำแหน่ง Y ของนิ้วกลาง
                current_middle_finger_y = middle_finger_tip.y
                if last_middle_finger_y is not None:
                    delta_y = current_middle_finger_y - last_middle_finger_y

                    # คำนวณระยะห่างระหว่างนิ้วชี้และนิ้วกลางเพื่อส่งเป็น finger_distance
                    finger_distance = ((index_finger_tip.x - middle_finger_tip.x) ** 2 +
                                       (index_finger_tip.y - middle_finger_tip.y) ** 2) ** 0.5

                    # เรียกฟังก์ชัน scroll โดยส่ง delta_y และ finger_distance
                    cursor_control.scroll(delta_y, finger_distance)  # เลื่อนสกอร์ลตามการเปลี่ยนแปลงใน Y

                last_middle_finger_y = current_middle_finger_y  # เก็บค่าของ Y จากเฟรมนี้
            else:
                last_middle_finger_y = None

    # แสดงผล
    # cv2.imshow('Hand Tracking', frame)  # ลดการแสดงผลภาพเพื่อลดการใช้ทรัพยากร

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อกล้อง
cap.release()
cv2.destroyAllWindows()