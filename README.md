# Panorama 全景影像拼接


<table>
   <tr>
      <td align="center"><b>原圖1</b><br><img src="HW2/01.JPG" width="350"/></td>
      <td align="center"><b>原圖2</b><br><img src="HW2/02.JPG" width="350"/></td>
   </tr>
   <tr>
      <td colspan="2" align="center"><b>拼接結果</b><br><img src="HW2/Stitched_opencv.jpg" width="1000"/></td>
   </tr>
</table>

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
目的是在影像中找到具有顯著特徵的點，這些點通常是角點或邊緣交叉處。

### Harris 角點檢測器
Harris 角點檢測器是一種基於影像梯度的角點檢測方法。其核心思想是利用影像在不同方向上的變化來識別角點。
1. **計算影像梯度**  
   使用 Sobel Filter 計算影像在 x 和 y 方向的梯度 (邊緣檢測)，分別得到 Ix 和 Iy。
   Sobel Filter 的卷積核如下：
   

   |              |                                 x 方向                                 |                                 y 方向                                 |
   | :----------: | :--------------------------------------------------------------------: | :--------------------------------------------------------------------: |
   | Sobel Kernel | $\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ | $\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$ |
   |    梯度圖    |                        ![Ix](HW2/grad_x_1.jpg)                         |                        ![Iy](HW2/grad_y_1.jpg)                         |
      - **x 方向 kernel**（Ix）：偵測豎線（垂直邊緣），對水平方向的變化較敏感。
   - **y 方向 kernel**（Iy）：偵測橫線（水平邊緣），對垂直方向的變化較敏感。

2. **計算 M 矩陣與角點響應值 R**  
   以每個像素為中心，計算其鄰域內的梯度乘積和，得到 Harris 矩陣 M：
   
   $M = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$
   
   接著計算角點響應值：
   
   $R = \det(M) - k \cdot (\operatorname{trace}(M))^2$
   
   其中 $k$ 通常取 0.04~0.06。
   


## Todo List
- [x] Harris Corner Detection
	- [ ] Switch to DOG
- [x] SIFT Descriptor
- [x] Feature Matching
- [ ] RANSAC
- [ ] Homography
- [ ] Image Warping
- [ ] Image Blending
- [ ] Documentation