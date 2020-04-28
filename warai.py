import cv2
import numpy

# カメラキャプチャ
cap = cv2.VideoCapture(0)
# 笑い男の動画かgifアニメ　背景が黒いやつ
gif = cv2.VideoCapture('icon.mp4')

# 分類器
# ここからダウンロード 
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

while True:
    # カメラから1フレームずつ取得
    ret, frame = cap.read()
    # フレームの反転
    frame = cv2.flip(frame, 1)

    # 笑い男アニメから1フレームずつ取得
    g, icon = gif.read()
    # ループ再生
    if not g:
        gif.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 笑い男アイコンのもともとの縦横比を計算
    orig_height, orig_width = icon.shape[:2]
    aspect_ratio = orig_width/orig_height

    # 顔検出
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facerect = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )
    
    if len(facerect) > 0:
        #検出した顔の数だけ処理を行う
        for rect in facerect:
            # 顔サイズに合わせて笑い男アイコンをリサイズ
            icon = cv2.resize(icon,tuple([int(rect[2]*aspect_ratio), int(rect[3])]))

            # 透過処理準備
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            icon = cv2.cvtColor(icon, cv2.COLOR_RGB2RGBA)

            # マスクの作成
            icon_gray = cv2.cvtColor(icon, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(icon_gray, 10, 255, cv2.THRESH_BINARY)

            # カメラフレームとリサイズ済み笑い男アイコンのサイズを取得
            height, width = icon.shape[:2]
            frame_height, frame_width = frame.shape[:2]

            # 合成時にはみ出さない場合だけ合成を行う
            if frame_height > rect[1]+height and frame_width > rect[0]+width:
                # 合成する座標を指定
                roi = frame[rect[1]:height+rect[1], rect[0]:width+rect[0]]

                # カメラフレームのうち、顔座標に相当する部分を笑い男アイコンに置き換える
                # マスクを使い、笑い男アイコン背景の黒い部分を透過させる
                frame[rect[1]:height+rect[1], rect[0]:width+rect[0]] = numpy.where(numpy.expand_dims(binary == 255, -1), icon, roi)

    cv2.imshow('result', frame)

    # 何らかのキーが入力されると終了
    k = cv2.waitKey(1)
    if k != -1:
        break

cap.release()
cv2.destroyAllWindows()
