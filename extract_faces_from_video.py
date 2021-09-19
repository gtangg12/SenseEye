import time
import numpy as np
import cv2
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1

from face_db import Person, FaceDB


video_path = 'george1.mp4'


class Face:
    def __init__(self, frame_number, crop):
        self.frame_number = frame_number
        self.crop = crop
        self.embedding = None
        self.person = None


def batch_eval(input_data, model, batch_size=100):
    output_data = []
    for idx in range(len(input_data) // batch_size + 1):
        output_data.extend(
            model(input_data[idx * batch_size : (idx + 1) * batch_size])
        )

    return output_data


def detect_faces(frames):
    mtcnn = MTCNN(image_size=224,
                  margin=32,
                  keep_all=True,
                  post_process=False,
                  thresholds=[0.6, 0.75, 0.9],
                  device='cuda:0')
    mtcnn.cuda()

    faces = []
    for frame_number, crops in enumerate(batch_eval(frames, mtcnn)):
        if crops is not None:
            for crop in crops:
                formatted_crop = np.transpose(crop.numpy(), (1, 2, 0)).astype(np.uint8)
                faces.append(Face(frame_number, formatted_crop))

    return faces


def compute_embeddings(faces):
    resnet = InceptionResnetV1(pretrained='vggface2').double().eval()
    resnet.cuda()

    crops = np.array([face.crop for face in faces])
    crops = (crops - 127.5) / 128.0
    crops = torch.tensor(np.transpose(crops, (0, 3, 1, 2)), device='cuda:0')

    for face, embedding in zip(faces, batch_eval(crops, resnet)):
        face.embedding = embedding.detach().cpu().numpy()


def tag_faces(faces, face_db):
    for face in faces:
        face.person = face_db.search_face(face)


def build_face_db(reference_data):
    face_db = FaceDB()

    for person, frames in reference_data.items():
        faces = detect_faces(frames[:10])
        compute_embeddings(faces)
        for face in faces:
            face_db.update_person(face, person)
    '''
    for person, index in face_db.person_index.items():
        print(person.name)
        print(face_db.average_embeddings[index][-10:])
    print()
    print()
    '''
    return face_db


def extract_video_frames(video_path):
    video_cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        success, frame = video_cap.read()
        if not success:
            break
        frames.append(frame[:,:,::-1]) #rgb to make colab compatible

    return np.array(frames)


def main():
    #frames = extract_frames_from_video(video_path)
    #faces = detect_faces(frames[:25])
    np.set_printoptions(linewidth=10000000)

    reference_data = {
        Person('George'): extract_video_frames('reference_videos/george.mp4')[:20],
        Person('William'): extract_video_frames('reference_videos/william.mp4')[:20],
    }

    face_db = build_face_db(reference_data)

    start = time.time()

    frames = extract_video_frames(video_path)
    faces = detect_faces(frames[:20])
    compute_embeddings(faces)
    tag_faces(faces, face_db)

    print(time.time() - start)

    for face in faces:
        if face.person != None:
            print(face.person.name)
        else:
            print("FK")
        cv2.imshow('Frame', face.crop)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
