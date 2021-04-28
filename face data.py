import cv2
import numpy as np

cap = cv2.VideoCapture(0)
path = "C:\\Users\\This PC\\final knn\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(path)
face_data = []
skip = 0
face_section = np.zeros((100, 100), dtype="uint8")
dir_path = "C:\\Users\\This PC\\Desktop\\data"

name = input("Enter your name:")

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2]*f[3])

    # pick the last face (largest)
    for face in faces[-1:]:
        x, y, w, h = face

        # extract main face
        face_section = gray_frame[y:y+h, x:x+w]
        face_section = cv2.resize(face_section, (100, 100))
        cv2.putText(frame, name, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)
    cv2.imshow("Camera", frame)

    if skip % 10 == 0:
        face_data.append(face_section)
    skip += 1
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)


#save this data into file system
np.save(dir_path+'\\'+name+'.npy', face_data)
print("Dataset saved at : {}".format(dir_path + name + '.npy'))

cap.release()
cv2.destroyAllWindows()
