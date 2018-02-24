import cv2
import os
import matplotlib.pyplot as plt
# from ImageRecognition.preparing_database import get_images_list_from_databse

class FaceDetection:

    def __init__(self, _database_path, _dirpath_out, image_format, _size_of_face=None):
        self.database_path = _database_path
        self.dirpath_out = _dirpath_out
        if _size_of_face:
            self.size_of_face = _size_of_face
        else:
            # self.size_of_face = (101,101)
            self.size_of_face = self.calculate_avarage_size_of_face()

    def detect_face(self, image_path):
        test1 = cv2.imread(image_path)
        gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

        cv2.CascadeClassifier()
        lbp_face_cascade = cv2.CascadeClassifier('lpb_frontalface')
        faces = lbp_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
        # print('Faces found: ', len(faces))
        return faces

    def calculate_avarage_size_of_face(self):
        h_buffer = 0
        w_buffer = 0
        face_counter = 0
        for root, _, files in os.walk(self.database_path):
            print("ss")
            print(root)
            print(files)
            for f in files:
                faces = self.detect_face(os.path.join(root, f))
                if len(faces) == 1:
                    [_, _, h, w] = faces[0]
                    face_counter += 1
                    h_buffer += h
                    w_buffer += w
                else:
                    print("not correct amount of faces on the picture")
                    print(f)
                    break
        mean_h = h_buffer/face_counter
        mean_w = w_buffer/face_counter
        print(h_buffer)
        print(w_buffer)
        return (int(mean_h), int(mean_w))

    def detect_faces_in_database(self):
        for root, _, files in os.walk(self.database_path):
            print(root)
            print(files)
            faces_counter = 0
            for f in files:
                face = self.detect_face(os.path.join(root, f))
                faces_counter += len(face)
                if len(face) != 1:
                    path = os.path.join(root, f)
                    os.unlink(path)
                else:
                    img = cv2.imread(os.path.join(root, f))
                    [x, y, w, h] = face[0]
                    crop_img = img[y:y + h, x:x + w]
                    resize_image = cv2.resize(crop_img, self.size_of_face)
                    if not os.path.exists(os.path.join(root, "faces")):
                        os.makedirs(os.path.join(root, "faces"))
                    cv2.imwrite(os.path.join(root, "faces", f), resize_image)


    def mark_detected_face(image_path, faces):
        test1 = cv2.imread(image_path)
        for (x, y, w, h) in faces:
            cv2.rectangle(test1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(test1)
        plt.show()


def clean_empty_dirs(base_dir):
    for dir in os.listdir(base_dir):
        print(dir)
        if not os.listdir(os.path.join(base_dir, dir)):
             os.rmdir(os.path.join(base_dir, dir))

# clean_empty_dirs("../lwf")

# fd = FaceDetection("../lwf", "../faces",".jpg")
# print(fd.size_of_face)
# fd.detect_faces_in_database()
