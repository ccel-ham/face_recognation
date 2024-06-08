import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import IPython
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm


class Config:
    ROOT_FOLDER_PATH = "/content/drive/MyDrive/snap"
    DETECT_FACE_TARGET_IMAGE_FOLDER_PATH = f"{ROOT_FOLDER_PATH}/names"
    TARGET_IMAGE_FOLDER_PATH = "/content/drive/MyDrive/snap/SportsEvent"
    GPU = False


def get_averages(names, scores):
    score_dict = defaultdict(list)
    for person, score in zip(names, scores):
        score_dict[person].append(score)

    averages = {}
    for person, score in score_dict.items():
        averages[person] = np.mean(score)
    return averages


def judge_similarity_by_embeddings(
    known_embeddings, known_names, unknown_embeddings, threshold
):
    predicted_persons = []
    predicted_scores = []
    for embeddings in unknown_embeddings:
        scores = np.dot(embeddings, known_embeddings.T)
        scores = np.clip(scores, 0.0, None)

        averages = get_averages(known_names, scores)
        person = sorted(averages, key=lambda x: averages[x], reverse=True)[0]
        score = averages[person]
        if score > threshold:
            predicted_persons.append(person)
            predicted_scores.append(score)
        else:
            predicted_persons.append(None)
            predicted_scores.append(score)

    return predicted_persons, predicted_scores


def get_face_rectangle_info(name):
    PAINT_INFO_MAP = {
        "FaceA": {"exec": True, "color": (255, 0, 0), "bolid": 1},
        "FaceB": {"exec": True, "color": (255, 255, 0), "bolid": 1},
        None: {"exec": False, "color": (0, 0, 255), "bolid": 1},
        "else": {"exec": False, "color": (0, 255, 255), "bolid": 1},
    }
    if name in PAINT_INFO_MAP:
        return (
            PAINT_INFO_MAP[name]["exec"],
            PAINT_INFO_MAP[name]["color"],
            PAINT_INFO_MAP[name]["bolid"],
        )
    else:
        return (
            PAINT_INFO_MAP["else"]["exec"],
            PAINT_INFO_MAP["else"]["color"],
            PAINT_INFO_MAP["else"]["bolid"],
        )


def draw_faces_with_rectangles(image, faces, name, score):
    drawn_image = image.copy()
    for i in range(len(faces)):
        face = faces[i]
        bounding_box = face.bbox.astype(int)
        flag, color, bolid = get_face_rectangle_info(name[i])
        if flag:
            cv2.rectangle(
                drawn_image,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[2], bounding_box[3]),
                color,
                bolid,
            )
            cv2.putText(
                drawn_image,
                f"{name[i]} {int(score[i])}",
                (bounding_box[0] - 1, bounding_box[1] - 4),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color,
                bolid,
            )

    return drawn_image


def get_target_embedding():
    known_names = []
    known_embeddings = []
    players = get_list(Path(Config.DETECT_FACE_TARGET_IMAGE_FOLDER_PATH))
    app_pre = mute_call_liblary()

    for player in tqdm(players):
        player_embeddings, player_names = [], []
        img_files = get_list(
            Path(f"{Config.DETECT_FACE_TARGET_IMAGE_FOLDER_PATH}/{player}"),
            get_file=True,
        )
        for file in img_files:
            img = cv2.imread(
                f"{Config.DETECT_FACE_TARGET_IMAGE_FOLDER_PATH}/{player}/{file}"
            )
            if img is None:
                continue

            faces = app_pre.get(np.array(img))
            if len(faces) == 0:
                continue
            player_embeddings.append(faces[0].embedding)
            player_names.append(player)

        player_embeddings = np.stack(player_embeddings, axis=0)
        known_embeddings.append(player_embeddings)
        known_names += player_names
    known_embeddings = np.concatenate(known_embeddings, axis=0)

    return known_embeddings, known_names


def mute_call_liblary():
    PROVIDERS = ["AzureExecutionProvider", "CPUExecutionProvider"]
    CTX = [1, 0]
    providers = ""
    ctx = 0
    if Config.GPU:
        providers = PROVIDERS[0]
        ctx = CTX[0]
    else:
        providers = PROVIDERS[1]
        ctx = CTX[1]

    with redirect_stdout(open(os.devnull, "w")):
        app = FaceAnalysis(providers=[providers])
        app.prepare(ctx_id=ctx, det_size=(640, 640))
    return app


def generate_random_digits(num_digits):
    if num_digits <= 0:
        num_digits = 1

    min_value = 10 ** (num_digits - 1)
    max_value = (10**num_digits) - 1
    random_number = random.randint(min_value, max_value)
    return random_number


def detect_exec(app, img_path, known_embeddings, known_names):
    test_img = cv2.imread(str(img_path))
    faces = app.get(np.array(test_img))
    unknown_embeddings = []
    for i in range(len(faces)):
        unknown_embeddings.append(faces[i].embedding)

    pred_names, pred_scores = judge_similarity_by_embeddings(
        known_embeddings, known_names, unknown_embeddings, 200
    )
    if has_target(pred_names):
        detect = draw_faces_with_rectangles(test_img, faces, pred_names, pred_scores)
        file_path = Path(
            f"{Config.TARGET_IMAGE_FOLDER_PATH}/recommend/{img_path.parent.name}/{img_path.stem}{img_path.suffix}"
        )
        folder_exists_and_make(file_path)
        cv2.imwrite(str(file_path), detect)


def get_list(path, get_file=False):
    if get_file:
        return [f.name for f in path.iterdir() if f.is_file()]
    else:
        return [f.name for f in path.iterdir() if f.is_dir()]


def folder_exists_and_make(folder_path):
    folder = folder_path.parent
    if not folder.exists():
        folder.mkdir(parents=True)


def has_target(pred_names):
    TARGET = ["FaceA", "FaceB"]
    return len(set(TARGET) & set(pred_names)) > 0


def concurrent_processing():
    known_embeddings, known_names = get_target_embedding()
    app = mute_call_liblary()
    subfolders = get_list(Path(Config.TARGET_IMAGE_FOLDER_PATH))
    for folder in subfolders:
        file_names = get_list(
            Path(f"{Config.TARGET_IMAGE_FOLDER_PATH}/{folder}"), get_file=True
        )
        print(f"folder {folder} - {len(file_names)} 件")
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = tqdm(
                (
                    detect_exec(
                        app,
                        Path(f"{Config.TARGET_IMAGE_FOLDER_PATH}/{folder}/{file_name}"),
                        known_embeddings,
                        known_names,
                    )
                    for file_name in file_names
                ),
                total=len(file_names),
            )
            for _ in results:
                pass


def main():
    start_time = time.time()
    if Config.TARGET_IMAGE_FOLDER_PATH:
        concurrent_processing()
    else:
        print(
            "Not set target image folder , input here --> [ Config.TARGET_IMAGE_FOLDER_PATH ]"
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n処理時間: {int(elapsed_time)}秒")


if __name__ == "__main__":
    main()
