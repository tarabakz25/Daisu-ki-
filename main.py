import cv2
import numpy as np

# 変形パラメータ (trackbar で操作するのでグローバルに置いておく)
params = {
    "tx": 0,
    "ty": 0,
    # スケールは trackbar 上で 1～20 → 実際は 0.1～2.0 にマッピング
    "sx": 10,
    "sy": 10,
    # 回転角度 (度数法): 0～360
    "angle": 0,
    # シアー (せん断) は trackbar 上で 0～20 → 実際は -10～+10 → -1.0～+1.0
    "shx": 10,
    "shy": 10
}

def create_affine_matrix(tx=0, ty=0, sx=1.0, sy=1.0, angle=0.0, shx=0.0, shy=0.0):
    """
    アフィン変換行列(3x3)を生成
    """
    # 回転 (ラジアン)
    rad = np.deg2rad(angle)
    cos_ = np.cos(rad)
    sin_ = np.sin(rad)

    # 回転行列
    R = np.array([
        [cos_, -sin_, 0],
        [sin_,  cos_, 0],
        [   0,     0, 1]
    ], dtype=np.float32)

    # 拡大行列
    S = np.array([
        [sx,  0,  0],
        [0,  sy,  0],
        [0,   0,  1]
    ], dtype=np.float32)

    # 平行移動行列
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]
    ], dtype=np.float32)

    # シアー(せん断)行列
    Sh = np.array([
        [1,   shx, 0],
        [shy, 1,   0],
        [0,    0,  1]
    ], dtype=np.float32)

    # 合成順序は拡大→回転→シアー→平行移動 (例)
    M_3x3 = T @ Sh @ R @ S
    return M_3x3

def transform_image_autosize(img, M_3x3):
    """
    変換後に画像がはみ出さないように自動で出力サイズを決定し、アフィン変換する
    """
    h, w = img.shape[:2]

    # 元画像の4隅 (x,y)
    corners = np.array([
        [0,   0],
        [w-1, 0],
        [0,   h-1],
        [w-1, h-1]
    ], dtype=np.float32)
    ones = np.ones((4,1), dtype=np.float32)
    corners_hom = np.hstack([corners, ones])  # shape: (4,3)

    # 4隅を M_3x3 で変換
    transformed_hom = (M_3x3 @ corners_hom.T).T  # shape: (4,3)
    # (x', y') を取り出し
    transformed_xy = transformed_hom[:, :2]

    # x, y の min, max
    min_x = np.min(transformed_xy[:,0])
    max_x = np.max(transformed_xy[:,0])
    min_y = np.min(transformed_xy[:,1])
    max_y = np.max(transformed_xy[:,1])

    width_new = int(np.ceil(max_x - min_x + 1))
    height_new = int(np.ceil(max_y - min_y + 1))

    # 画像が負座標に行かないように補正
    # 左上が(0,0)になるようにだけ平行移動
    M_adjust = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0,  1     ]
    ], dtype=np.float32)

    M_final = M_adjust @ M_3x3  # 最終合成行列
    M_2x3 = M_final[:2, :3]

    # warpAffine
    # (width_new, height_new) ははみ出しがないように自動計算したサイズ
    transformed = cv2.warpAffine(img, M_2x3, (width_new, height_new))
    return transformed

def resize_for_display(img, max_w=1280, max_h=800):
    """
    画像が大きい場合、表示用に縮小する
    """
    h, w = img.shape[:2]
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# ---- 以下、Trackbarのコールバック ----

def on_change_tx(v):
    # trackbar: 0~1000 → 実際: -500~+500
    params["tx"] = v - 500

def on_change_ty(v):
    params["ty"] = v - 500

def on_change_sx(v):
    # 1~20 → 0.1~2.0
    if v < 1: v = 1
    params["sx"] = v / 10.0

def on_change_sy(v):
    # 1~20 → 0.1~2.0
    if v < 1: v = 1
    params["sy"] = v / 10.0

def on_change_angle(v):
    # 0~360
    params["angle"] = v

def on_change_shx(v):
    # 0~20 → -10~+10 → -1.0~+1.0
    params["shx"] = (v - 10) / 10.0

def on_change_shy(v):
    params["shy"] = (v - 10) / 10.0

def main():
    img_path = "test.png"  # 適宜変更
    img = cv2.imread(img_path)
    if img is None:
        print("画像が読み込めませんでした:", img_path)
        return

    # ウィンドウ作成
    cv2.namedWindow("FreeTransform", cv2.WINDOW_NORMAL)

    # トラックバーを作成
    cv2.createTrackbar("TX", "FreeTransform", 500, 1000, on_change_tx)  # -500~+500
    cv2.createTrackbar("TY", "FreeTransform", 500, 1000, on_change_ty)

    cv2.createTrackbar("SX", "FreeTransform", 10, 20, on_change_sx)    # 1~20 => 0.1~2.0
    cv2.createTrackbar("SY", "FreeTransform", 10, 20, on_change_sy)

    cv2.createTrackbar("Angle", "FreeTransform", 0, 360, on_change_angle)

    cv2.createTrackbar("SHX", "FreeTransform", 10, 20, on_change_shx)  # 0~20 => -1.0~+1.0
    cv2.createTrackbar("SHY", "FreeTransform", 10, 20, on_change_shy)

    print("[操作説明]")
    print("- トラックバーで平行移動(tx, ty), 拡大率(sx, sy), 回転(angle), シアー(shx, shy)を変更")
    print("- [q]キーで終了, [s]キーで現在の結果画像を保存 (transform_result.jpg)")

    while True:
        # 現在のパラメータからアフィン行列を作る
        M_3x3 = create_affine_matrix(
            tx=params["tx"],
            ty=params["ty"],
            sx=params["sx"],
            sy=params["sy"],
            angle=params["angle"],
            shx=params["shx"],
            shy=params["shy"]
        )

        # はみ出しを防ぐ自動リサイズ版 transform
        transformed_full = transform_image_autosize(img, M_3x3)
        
        # 表示用に縮小（画面からはみ出さないように）
        display_img = resize_for_display(transformed_full, max_w=1280, max_h=800)
        
        cv2.imshow("FreeTransform", display_img)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # フルサイズで保存
            cv2.imwrite("transform_result.jpg", transformed_full)
            print("画像を保存しました: transform_result.jpg")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
