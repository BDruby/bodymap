import cv2
import mediapipe as mp
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

pTime = 0
cTime = 0

wSrc, hSrc = pyautogui.size()
wCap, hCap = 1280, 720
# 改下样式
facePoStyle = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
faceConStyle = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)

posePoStyle = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=5)
poseConStyle = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)

# 摄像头
cap = cv2.VideoCapture(0)
cap.set(3, wCap)
cap.set(4, hCap)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        # 逐帧图片RGB化
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 水平翻转解决左右相反的镜像问题
        image = cv2.flip(image, flipCode=1)
        # 逐帧检测图片
        result= holistic.process(image)
        # 印出关键点坐标测试
        # print(result.face_landmarks)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgH = image.shape[0]
        imgW = image.shape[1]
        # 脸部
        mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS, facePoStyle, faceConStyle)
        # 姿势
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, posePoStyle, poseConStyle)
        # 右手
        mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # 左手
        mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        imgH = image.shape[0]
        imgW = image.shape[1]
        
        #实现点击的方法，虽然上面左右翻转过，但获取到的图像数据依然左右相反才对，所以left是右手，right是左手
        if result.left_hand_landmarks:
            for i, landmarks in enumerate(result.left_hand_landmarks.landmark):
                # print(i, xPos, yPos)
                if i == 8:
                    xPot8 = int(landmarks.x * wSrc)
                    yPot8 = int(landmarks.y * hSrc)
                    cv2.putText(image, str('Shi'), (xPot8 - 25, yPot8 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255))
                    pyautogui.moveTo(xPot8, yPot8)
                if i == 12:
                    xPot12 = int(landmarks.x * wSrc)
                    yPot12 = int(landmarks.y * hSrc)
                    cv2.putText(image, str('Zhong'), (xPot12 - 25, yPot12 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255))
                    if yPot8 - yPot12 > 0:
                        pyautogui.click()

        # 左上角帧率
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(image, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow('img', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
