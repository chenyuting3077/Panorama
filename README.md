# Panorama 全景影像拼接

## 簡介
全景影像（Panorama）常用於捕捉完整場景，尤其是在單張照片無法涵蓋整個景觀時。本專案透過 **影像拼接（Image Stitching）** 技術，將多張相同場景的影像合成一張完整的全景圖。  

整個流程可分為五個主要階段：
1. **特徵點檢測（Detect Keypoints）**  
   - 使用 **Harris 角點檢測器** 找到影像中重要的角點。  
   - 將 RGB 影像轉為灰階，並透過 Sobel Filter 計算影像梯度。
   
2. **特徵點描述（Describe Keypoints）**  
   - 使用 **SIFT 描述子** 將每個角點的局部梯度訊息編碼成向量。  
	- (Note: SIFT 原先使用 DOG 找特徵點)
   - 生成 128 維 descriptor，使特徵點具有旋轉與尺度不變性。
   
3. **特徵點匹配（Match Keypoints）**  
   - 比較兩張影像的 SIFT descriptor，透過 L2 距離找到匹配對。  
   - 設定閾值過濾不可靠的匹配。

4. **匹配評估（Evaluate Correspondences）**  
   - 使用 **RANSAC** 篩選出最穩定的匹配對。  
   - 根據匹配點計算 **單應性矩陣（Homography）**，用於影像對齊。

5. **影像變換與融合（Transform & Blend）**  
   - 利用單應性矩陣將一張影像投影到另一張影像的平面上。  
   - 對重疊區域進行融合，使拼接後的全景圖自然連貫。


## 特徵點檢測（Detect Keypoints)
特徵點檢測是影像拼接的第一步，目的是在影像中找到具有顯著特徵的點，這些點通常是角點或邊緣交叉處。

### Harris 角點檢測器
Harris 角點檢測器是一種基於影像梯度的角點檢測方法。其核心思想是利用影像在不同方向上的變化來識別角點。
1. **計算影像梯度**  
   使用 Sobel Filter 計算影像在 x 和 y 方向的梯度 (邊緣檢測)，分別得到 Ix 和 Iy。
   




## Todo List
- [x] Harris Corner Detection
	- [ ] Switch to DOG
- [x] SIFT Descriptor
- [x] Feature Matching
- [x] RANSAC
- [ ] Homography
- [ ] Image Warping
- [ ] Image Blending
- [ ] Documentation