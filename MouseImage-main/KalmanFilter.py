import numpy as np

class KalmanFilter:
    def __init__(self):
        # กำหนดสถานะเริ่มต้นของ Kalman Filter
        self.state = np.zeros(4)  # [x, y, vx, vy] ตำแหน่งและความเร็ว
        self.P = np.eye(4) * 1000  # เมทริกซ์ความแปรปรวนเริ่มต้นด้วยค่ามาก
        self.F = np.array([[1, 0, 1, 0],  # เมทริกซ์การเปลี่ยนสถานะ
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],  # เมทริกซ์การวัด
                           [0, 1, 0, 0]])

        # ปรับค่าความไม่แน่นอนของการวัดและกระบวนการ
        self.R = np.eye(2) * 0.3  # ลดค่าความไม่แน่นอนของการวัดเพื่อลดความหน่วง
        self.Q = np.array([[0.2, 0, 0, 0],  # เพิ่มค่าความไม่แน่นอนของกระบวนการ
                           [0, 0.2, 0, 0],
                           [0, 0, 0.1, 0],
                           [0, 0, 0, 0.1]])

    def predict(self):
        # คำนวณสถานะใหม่ด้วยการคาดการณ์
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        # คำนวณการแก้ไขสถานะตามการวัดใหม่
        y = measurement - np.dot(self.H, self.state)  # ความแตกต่างของการวัด
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # ความแปรปรวนของความแตกต่าง
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # ค่ากำไรของ Kalman
        self.state = self.state + np.dot(K, y)  # อัปเดตสถานะ
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # อัปเดตเมทริกซ์ความแปรปรวน

    def get_state(self):
        # คืนค่าตำแหน่งที่กรองแล้ว
        return self.state[0], self.state[1]
