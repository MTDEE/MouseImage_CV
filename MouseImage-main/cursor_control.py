import pyautogui

class CursorControl:
    def __init__(self, screen_width, screen_height, scroll_speed=30, sensitivity=190):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scroll_speed = scroll_speed
        self.sensitivity = sensitivity
        self.last_y_position = None
        self.is_scrolling = False  # ตัวแปรเช็คว่ากำลังเลื่อนหรือไม่
        self.last_finger_distance = 0  # เก็บค่าระยะห่างของนิ้วที่ใช้เลื่อน

    def move_cursor(self, x, y):
        pyautogui.moveTo(x, y)

    def click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def scroll(self, delta_y, finger_distance):
        """
        เลื่อนหน้าจอเมื่อมีการเลื่อนนิ้ว
        - delta_y: ความแตกต่างในแนวตั้งระหว่างตำแหน่งใหม่และตำแหน่งเดิมของนิ้ว
        - finger_distance: ระยะห่างระหว่างนิ้วชี้และนิ้วกลาง
        """
        if abs(delta_y) > 0.01:
            # บันทึกตำแหน่งเมาส์ก่อนเริ่มการเลื่อน
            current_mouse_x, current_mouse_y = pyautogui.position()

            if finger_distance != self.last_finger_distance:  # ถ้าระยะห่างของนิ้วเปลี่ยนแปลง
                scroll_amount = int(delta_y * self.scroll_speed * self.sensitivity)
                pyautogui.scroll(scroll_amount)

                # กลับไปที่ตำแหน่งเดิมหลังจากการเลื่อน
                pyautogui.moveTo(current_mouse_x, current_mouse_y)

                self.last_finger_distance = finger_distance  # อัปเดตระยะห่างของนิ้ว
                self.is_scrolling = True
            else:
                self.is_scrolling = False  # เมื่อระยะห่างไม่เปลี่ยนแปลง ให้หยุดเลื่อน

    def stop_scroll(self):
        """
        หยุดการเลื่อนเมื่อไม่มีการเคลื่อนไหวของนิ้วชี้และนิ้วกลาง
        """
        if not self.is_scrolling:
            self.last_finger_distance = 0  # รีเซ็ตระยะห่างของนิ้ว
            self.last_y_position = None  # รีเซ็ตค่าตำแหน่ง Y ของนิ้วกลางเพื่อหยุดการเลื่อน

# การทดสอบในส่วนของ main loop
cursor_control = CursorControl(1920, 1080)

# กำหนดค่า finger_distance ก่อน
finger_distance = 50  # ตัวอย่างค่า
delta_y = 10  # ตัวอย่างค่า
# ส่งทั้งสองพารามิเตอร์
cursor_control.scroll(delta_y, finger_distance)