import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
    ret, frame = cap.read()
    print(f"Camera {i}: {'Working' if ret else 'Not working'}")
    cap.release()