<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat" data-canonical-src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&amp;style=for-the-badge" srcset="https://qiita-user-contents.imgix.net/https%3A%2F%2Fimg.shields.io%2Fbadge%2F-Python-F2C63C.svg%3Flogo%3Dpython%26style%3Dfor-the-badge?ixlib=rb-4.0.0&amp;auto=format&amp;gif-q=60&amp;q=75&amp;w=1400&amp;fit=max&amp;s=5d7d909c2f70c6c8a0fc0477bd1a56ae 1x" loading="lazy">
# 概要
このリポジトリには、insightfaceを使用して顔認識を行うPythonスクリプトが含まれています。  
このスクリプトは画像内の顔を検出し、既知の顔と照合し認識された顔の周囲に境界ボックスを描画・名前を表示します。  

# 使用ライブラリ
**Python 3.x**  
**NumPy**: 数値計算において基盤となるデータ操作を行うために使用します。  
**OpenCV (cv2)**: 画像の読み込み、四角形の描画など画像処理に使用します。  
**Insightface**: 顔認識や特徴量の抽出に使用します。  
**tqdm**: Pythonのプログレスバーライブラリです。処理の進捗状況をユーザーに示すために使用します。  

# フォルダ構成
<pre>
teacher data  
names ── faceA ── img1.jpg  
　　　 │　　　　 └─ img2.jpg  
　　　 ├─ faceB ── img1.jpg  
　　　 │　　　　 └─ img2.jpg  
　　　 └── faceC ── img1.jpg   


detect data  
main_folde ── sub_folderA ─── img1.jpg  
　　　　　　│　　　　　　    ├─ img2.jpg  
　　　　　　│　　　　　　    ├─ img3.jpg  
　　　　　　│　　　　　　    └─ img4.jpg  
　　　　　　├─ sub_folderB ──── img1.jpg  
　　　　　　|　　　　　　　  └── img2.jpg  
　　　　　　├─ sub_folderC ─── img1.jpg  

</pre>

# 使用方法
・リポジトリをローカルマシンにクローンします。  
・必要なPythonライブラリがインストールされていることを確認します。pipを使用してインストールできます。  
　→　pip install -r requirements.txt  
・スクリプトの先頭にある Config クラスの構成パラメータを更新して環境を設定します。  
　BASE_PATH：対象の画像と対象名が格納されているベースパス。  
　PLAYER_PATH：対象名が含まれるフォルダのパス。  
　IMAGE_PATH：処理する画像が含まれるフォルダのパス。  
　GPU：顔認識にGPUを使用する場合は True に設定し、それ以外の場合は False に設定します。  
・対象の画像と対象名を準備します：  
　個々の対象（対象名）の名前でフォルダを作成し、その中に対象の画像を整理します。  
　これらのフォルダを DETECT_FACE_TARGET_IMAGE_FOLDER_PATH で指定されたディレクトリに配置します。  
・main() 関数を実行します。  
・スクリプトはTARGET_IMAGE_FOLDER_PATH内の画像に対して顔認識を実行し、認識された顔の周囲に境界ボックスを描画し、処理された画像をIMAGE_PATH内にrecomendフォルダーを作成して保存します。  

# 追加の注意事項
・このスクリプトは、ThreadPoolExecutorを使用した並行処理をサポートしており、より高速な実行が可能です。  
・顔認識の閾値は200に設定されています。必要に応じて、judge_sim関数でこの閾値を調整できます。  
・画像に描画するための色と境界ボックスの太さは、 get_paint 関数で定義されています。
　必要に応じてこれらの設定をカスタマイズできます。  
