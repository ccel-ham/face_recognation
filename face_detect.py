import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from glob import glob
import IPython
from insightface.app import FaceAnalysis
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm


class Config:
    BASE_PATH = "/content/drive/MyDrive/snap"
    PLAYER_PATH = f"{BASE_PATH}/names"
    IMAGE_PATH = "/content/drive/MyDrive/snap/SportsEvent"
    GPU = False


def get_averages(names, scores):
    d = defaultdict(list)
    for n, s in zip(names, scores):
        d[n].append(s)

    averages = {}
    for n, s in d.items():
        averages[n] = np.mean(s)
    return averages


def judge_sim(known_embeddings, known_names, unknown_embeddings, threshold):
    pred_names = []
    pred_scores = []
    for emb in unknown_embeddings:
        scores = np.dot(emb, known_embeddings.T)
        scores = np.clip(scores, 0., None)

        averages = get_averages(known_names, scores)
        pred = sorted(averages, key=lambda x: averages[x], reverse=True)[0]
        score = averages[pred]
        if score > threshold:
            pred_names.append(pred)
            pred_scores.append(score)
        else:
            pred_names.append(None)
            pred_scores.append(score)

    return pred_names, pred_scores

def get_paint(name):
    COLOR_MAP = {"FaceA":{"exec":True,
                        "color":(255,0,0),
                        "bolid":1},
                "FaceB":{"exec":True,
                        "color":(255,255,0),
                        "bolid":1},
                None:{"exec":False,
                        "color":(0,0,255),
                        "bolid":1},
                "else":{"exec":False,
                        "color":(0,255,255),
                        "bolid":1},
                }
    _tmp = COLOR_MAP.get(name)
    if not _tmp:
        return  COLOR_MAP["else"]["exec"], COLOR_MAP["else"]["color"], COLOR_MAP["else"]["bolid"]
    else:
        return COLOR_MAP[name]["exec"], COLOR_MAP[name]["color"], COLOR_MAP[name]["bolid"]


def draw_on(img, faces, name, score):
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(int)
        flag, color, bolid = get_paint(name[i])
        if flag:
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, bolid)
            cv2.putText(dimg, f"{name[i]} {int(score[i])}", (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.5,color,bolid)

    return dimg


def get_target_embedding():
    known_names = []
    known_embeddings = []

    players = get_list(Path(Config.PLAYER_PATH))
    app_pre = mute_call_liblary()

    for player in tqdm(players):
        player_embeddings, player_names = [], []
        img_files = get_list(Path(f"{Config.PLAYER_PATH}/{player}"), get_file=True)
        for file in img_files:
            img = cv2.imread(f"{Config.PLAYER_PATH}/{player}/{file}")
            if img is None: continue

            faces = app_pre.get(np.array(img))
            if len(faces) == 0 : continue
            player_embeddings.append(faces[0].embedding)
            player_names.append(player)

        player_embeddings = np.stack(player_embeddings, axis=0)
        known_embeddings.append(player_embeddings)
        known_names += player_names
    known_embeddings = np.concatenate(known_embeddings, axis=0)

    return known_embeddings, known_names


def mute_call_liblary():
    PROVIDERS = ['AzureExecutionProvider', 'CPUExecutionProvider']
    CTX = [0, 1]
    providers = ""
    ctx = 0
    if Config.GPU :
        providers = PROVIDERS[0]
        ctx = CTX[0]
    else:
        providers = PROVIDERS[1]
        ctx = CTX[1]

    with redirect_stdout(open(os.devnull, "w")):
        app = FaceAnalysis(providers = [providers])
        app.prepare(ctx_id = ctx, det_size = (640, 640))
    return app


def detect_exec(app, img_path, known_embeddings, known_names):
    test_img = cv2.imread(str(img_path))
    faces = app.get(np.array(test_img))
    unknown_embeddings = []
    for i in range(len(faces)):
        unknown_embeddings.append(faces[i].embedding)

    pred_names, pred_scores = judge_sim(known_embeddings, known_names, unknown_embeddings, 200)
    if has_target(pred_names):
        detect = draw_on(test_img, faces, pred_names, pred_scores)
        file_path = Path(f"{Config.IMAGE_PATH}/recommend/{img_path.parent.name}/{img_path.stem}{img_path.suffix}")
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
    TARGET =["FaceA", "FaceB"]
    return len(set(TARGET) & set(pred_names)) > 0


def concurrent_processing():

    known_embeddings, known_names = get_target_embedding()
    app = mute_call_liblary()
    subfolders = get_list(Path(Config.IMAGE_PATH))
    for folder in subfolders:
        file_names = get_list(Path(f"{Config.IMAGE_PATH}/{folder}"), get_file=True)
        print(f"folder {folder} - {len(file_names)} 件")
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = tqdm((detect_exec(app, Path(f"{Config.IMAGE_PATH}/{folder}/{file_name}"), known_embeddings, known_names) for file_name in file_names), total=len(file_names))
            for _ in results:
                pass

def main():
    start_time = time.time()
    if Config.IMAGE_PATH:
        concurrent_processing()
    else:
        print("Not set target image folder , input here --> [ Config.IMAGE_PATH ]")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n処理時間: {int(elapsed_time)}秒")

if __name__=="__main__":
  main()