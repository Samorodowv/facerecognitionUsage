import os
import shelve

import cv2
import face_recognition as fr
import numpy as np


def print_help():
    print("""
    Для первого использования необходимо положить фотографии в папку faces,
    в качестве имен файлов используя имена людей на фото.
    При необходимости повторно обучить детектор, необходимо удалить файлы facedata
    """)


def make_data():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                print(f"LOADING {len(encoded)}: {f}")
                try:
                    encoding = fr.face_encodings(face)[0]
                    encoded[f.split(".")[0]] = encoding
                except Exception as e:
                    print(e)
                    pass
    if len(encoded) == 0:
        print_help()
        exit()

    with shelve.open('facedata.db') as facedata:
        facedata['facedata'] = encoded
    return encoded


def restore_data():
    with shelve.open('facedata.db') as facedata:
        return facedata.get('facedata')


if __name__ == "__main__":
    data = restore_data()
    if data is None:
        data = make_data()

    face_encodings = list(data.values())
    known_face_names = list(data.keys())

    cap = cv2.VideoCapture(0)

    # обработка каждого второго кадра для оптимизации
    process_frames = True
    face_locations = []
    unknown_face_encodings = []
    face_names = []
    while True:
        success, frame = cap.read()
        if cv2.waitKey(1) == 27 or not success:
            break
        img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        img = img[:, :, ::-1]
        if process_frames:
            face_locations = fr.face_locations(img)
            unknown_face_encodings = fr.face_encodings(img, face_locations)
            face_names = []
            for face_encoding in unknown_face_encodings:
                matches = fr.compare_faces(face_encodings, face_encoding)
                name = "Unknown"
                face_distances = fr.face_distance(face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[int(best_match_index)]
                face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left - 20, top - 20),
                          (right + 20, bottom + 20), (0, 128, 255), 2)

            cv2.rectangle(frame, (left - 20, bottom - 15),
                          (right + 20, bottom + 20), (0, 128, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, name.lower().title(), (left - 20,
                                                      bottom + 15), font, 0.5, (255, 255, 255), 1)

        process_frames = not process_frames
        cv2.imshow('result', frame)

    print("Выполнение программы приостановлено")
    cap.release()
    cv2.destroyAllWindows()
    exit()
