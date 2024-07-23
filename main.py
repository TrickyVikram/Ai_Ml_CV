import cv2
import face_recognition

def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    print(f"Loaded image: {image_path}, Type: {type(image)}, Shape: {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

try:
    imgModi = load_and_convert_image('Images_Attendance/modi-image-for-InUth.jpg')
    imgTest = load_and_convert_image('Images_Attendance/narendra-modi.jpg')

    faceloc = face_recognition.face_locations(imgModi)[0]
    encodeModi = face_recognition.face_encodings(imgModi)[0]
    cv2.rectangle(imgModi, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)

    facelocTest = face_recognition.face_locations(imgTest)[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]  # Change index from [1] to [0]
    cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (155, 0, 255), 2)

    results = face_recognition.compare_faces([encodeModi], encodeTest)
    faceDis = face_recognition.face_distance([encodeModi], encodeTest)
    print(results, faceDis)

    cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow('modi', imgModi)
    cv2.imshow('narendra-modi', imgTest)
    cv2.waitKey(0)  # Change from waitKeys(0) to waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")
