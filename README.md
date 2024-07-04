# face_recognation  
face recognation system  

# 概要
このリポジトリには、insightfaceを使用して顔認識を行うPythonスクリプトが含まれています。このスクリプトは、画像内の顔を検出し、既知の顔と照合し、認識された顔の周囲に境界ボックスを描画し、名前を表示します。  

# 使用ライブラリ
**Python 3.x**
**NumPy**
**OpenCV (cv2)**
**Insightface**
**tqdm**
**Pillow**  

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
・pip install numpy opencv-python insightface tqdm pillow  
・スクリプトの先頭にある Config クラスの構成パラメータを更新して環境を設定します：  
　BASE_PATH：対象の画像と対象名が格納されているベースパス。  
　PLAYER_PATH：対象名が含まれるフォルダのパス。  
　IMAGE_PATH：処理する画像が含まれるフォルダのパス。  
　GPU：顔認識にGPUを使用する場合は True に設定し、それ以外の場合は False に設定します。  
・対象の画像と対象名を準備します：  
　個々の対象（対象名）の名前でフォルダを作成し、その中に対象の画像を整理します。  
　これらのフォルダを PLAYER_PATH で指定されたディレクトリに配置します。  
・main() 関数を実行してスクリプトを実行します。  
・スクリプトは IMAGE_PATH 内の画像に対して顔認識を実行し、認識された顔の周囲に境界ボックスを描画し、処理された画像をIMAGE_PATH内にrecomendフォルダーを作成して保存します。  

# 追加の注意事項
・このスクリプトは、ThreadPoolExecutorを使用した並行処理をサポートしており、より高速な実行が可能です。  
・顔認識の閾値は200に設定されています。必要に応じて、 judge_sim 関数でこの閾値を調整できます。  
・画像に描画するための色と境界ボックスの太さは、 get_paint 関数で定義されています。必要に応じてこれらの設定をカスタマイズできます。  
  
