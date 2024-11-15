import cv2
import mediapipe as mp

class HandTracking:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
        # กำหนดค่า mp_hands ที่จะใช้ในทุกฟังก์ชั่น
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # แปลงภาพเป็น RGB และตรวจจับมือ
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    def draw_landmarks(self, frame, results):
        # วาด landmarks ของมือทุกๆ ข้อ
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def calculate_thumb_middle_distance(self, hand_landmarks):
        # คำนวณระยะห่างระหว่างนิ้วโป้งและนิ้วกลาง
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        distance = ((thumb_tip.x - middle_finger_tip.x) ** 2 + (thumb_tip.y - middle_finger_tip.y) ** 2) ** 0.5
        return distance

    def get_finger_positions(self, hand_landmarks):
        # ดึงตำแหน่งของนิ้วต่างๆ เช่น นิ้วโป้ง, นิ้วชี้, และนิ้วกลาง
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        return thumb_tip, index_finger_tip, middle_finger_tip

    def get_hand_center(self, hand_landmarks):
        # คำนวณตำแหน่งศูนย์กลางของมือ
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        palm_center = ((wrist.x + wrist.y) / 2)
        return palm_center