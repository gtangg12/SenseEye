import numpy as np

class Person:
    def __init__(self, name):
        self.name = name


class FaceDB:
    UNKNOWN_PERSON_THRESHOLD = 0.75

    def __init__(self):
        self.person_index = {}
        self.index_person = {}
        self.embeddings = []
        self.average_embeddings = np.zeros((10000, 512))
        self.size = 0

    def _add_embedding(self, embedding, index):
        self.embeddings[index].append(embedding)

        n = len(self.embeddings)
        self.average_embeddings[index] = \
            (self.average_embeddings[index] * (n - 1) + embedding) / n

    def update_person(self, face, person):
        if person not in self.person_index:
            index = self.size
            self.person_index[person], self.index_person[index] = index, person
            self.embeddings.append([])
            self.size += 1
        else:
            index = self.person_index[person]

        self._add_embedding(face.embedding, self.person_index[person])


    # assume face has already been identified
    def search_face(self, face, add=False):
        distances = np.linalg.norm(face.embedding - self.average_embeddings[:self.size], axis=1)
        index = np.argmin(distances)
        print(distances)
        if distances[index] > self.UNKNOWN_PERSON_THRESHOLD:
            return None
        if add:
            self.add_embedding(face.embedding, index)
        return self.index_person[index]

    def load_db(self, path):
        pass

    def save_db(self, path):
        pass


from sortedcontainers import SortedList

# assume all people in face_db
class FastFaceDB:
    CACHE_SIZE = 64

    def __init__(self, face_db):
        self.faces = face_db

        self.lru = {}
        self.timestamp = 0

    def update_person(self, face, person):
        if person not in face_db.person_index:
            return

        person_removed = None

        if person not in cache.person_index and len(self.lru) >= CACHE_SIZE:
            min_timestamp = 99999
            for person, (timestamp, _) in self.lru.items():
                if timestamp < min_timestamp:
                    min_timestamp = timestamp
                    person_removed = person

            self.lru.remove(person_removed)

        self.lru[person] = (
            timestamp,
            self.face_db.average_embeddings[self.face_db.person_index[person]]
        )
        
    def search_face(self, face):
        min_distance = 999999
        best_person = None

        for person, (_, average_embedding) in self.lru:
            distance = np.linalg.norm(face.embedding - average_embedding)
            if distance < min_distance:
                min_distance = distance
                best_person = person

        if min_distance < 0.5:
            return best_person
        return face_db.search_face(face)
