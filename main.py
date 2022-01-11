from time import sleep

import cv2
import cvzone
import numpy as np
from pynput.keyboard import Controller
import mediapipe as mp
import math as m

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

final_text = ""

keyboard = Controller()

# 寻找手部框架
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


def draw(screen, buttonList):
    for i in buttonList:
        button_x, button_y = i.pos
        button_w, button_h = i.size
        cvzone.cornerRect(screen, (i.pos[0], i.pos[1],
                                   i.size[0], i.size[0]), 20, rt=0)
        cv2.rectangle(screen, i.pos, (int(button_x + button_w), int(button_y + button_h)), (255, 144, 30), cv2.FILLED)
        cv2.putText(screen, i.text, (button_x + 20, button_y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
    return screen


def transparent_layout(screen, buttonList):
    imgNew = np.zeros_like(screen, np.uint8)
    for i in buttonList:
        x, y = i.pos
        cvzone.cornerRect(imgNew, (i.pos[0], i.pos[1],
                                   i.size[0], i.size[0]), 20, rt=0)
        cv2.rectangle(imgNew, i.pos, (x + i.size[0], y + i.size[1]),
                      (255, 144, 30), cv2.FILLED)
        cv2.putText(imgNew, i.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

    out = screen.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(screen, alpha, imgNew, 1 - alpha, 0)[mask]
    return out


class ButtonList:
    class Button:
        def __init__(self, pos, text, size=None):
            if size is None:
                size = [85, 85]
            self.pos = pos
            self.size = size
            self.text = text

    keyboard_keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                     ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                     ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
    buttonList = []

    for k in range(len(keyboard_keys)):
        for x, key in enumerate(keyboard_keys[k]):
            buttonList.append(Button([100 * x + 25, 100 * k + 50], key))


if __name__ == "__main__":
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # 画出手掌轮廓
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                    if id == 8:
                        x8, y8 = cx, cy
                    if id == 12:
                        x12, y12 = cx, cy
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for button in ButtonList.buttonList:
                x, y = button.pos
                w, h = button.size
                if x < x8 < x + w and y < y8 < y + h:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    long = m.sqrt((x8-x12)*(x8-x12)+(y8-y12)*(y8-y12))
                    if long < 50:
                        keyboard.press(button.text)
                        cv2.rectangle(img, button.pos, (x + w, y + h),
                                      (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                        final_text += button.text
                        sleep(0.2)

        img = transparent_layout(img, ButtonList.buttonList)
        cv2.rectangle(img, (25, 350), (700, 450),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(img, final_text, (60, 425),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
        cv2.imshow("output", img)
        cv2.waitKey(1)
