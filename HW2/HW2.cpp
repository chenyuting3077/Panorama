#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/features2d.hpp"
#include <cmath>
#include <vector>
#include <math.h>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <sstream>
#include <Eigen/Dense>
#include <random>
#include <numeric> 


using namespace std;
using namespace cv;
using namespace Eigen;
# define PI 3.14159265358979323846
# define ONE_EIGHTH_PI (1.0/4.0*PI)
const float ORIENTATION[9] = { 0, 1*ONE_EIGHTH_PI, 2* ONE_EIGHTH_PI , 3 * ONE_EIGHTH_PI , 4 * ONE_EIGHTH_PI , 5 * ONE_EIGHTH_PI , 6 * ONE_EIGHTH_PI , 7 * ONE_EIGHTH_PI , 8 * ONE_EIGHTH_PI };


/*====================  my_blur ===================*/
const Mat Gaussian_Filter = (Mat_<float>(3, 3) <<
    1, 2, 1,
    2, 4, 2,
    1, 2, 1) / 16.0f;

void my_blur(const Mat& src, Mat& dst) {
    // 在做 Sobel 前，對原始影像做高斯平滑（如 my_blur），目的是抑制雜訊。
    // Point(-1, -1) 表示 kernel 以中心點為 anchor（自動置中）
    // 0 表示不加偏移
    // BORDER_DEFAULT 表示邊界外的像素值由 OpenCV 自動決定
    filter2D(src, dst, -1, Gaussian_Filter, Point(-1, -1), 0, BORDER_DEFAULT);
    // 若要回傳原型態（如 CV_8U），可再轉回
}
/*=================== harris_detector ===============================*/
// 用 filter2D 計算 x 方向 Sobel 梯度
void my_sobel_x(const Mat& src, Mat& grad_x) {
    // SOBEL_FILTER_GX 為 x 方向 kernel
    const Mat SOBEL_FILTER_GX =
        (Mat_<float>(3, 3) <<
            +1.0, 0.0, -1.0,
            +2.0, 0.0, -2.0,
            +1.0, 0.0, -1.0);
    filter2D(src, grad_x, -1, SOBEL_FILTER_GX, Point(-1, -1), 0, BORDER_DEFAULT);
    
}

// 用 filter2D 計算 y 方向 Sobel 梯度
void my_sobel_y(const Mat& src, Mat& grad_y) {
    const Mat SOBEL_FILTER_GY =
        (Mat_<float>(3, 3) <<
            +1.0, +2.0, +1.0,
            +0.0, +0.0, +0.0,
            -1.0, -2.0, -1.0);
    filter2D(src, grad_y, -1, SOBEL_FILTER_GY, Point(-1, -1), 0, BORDER_DEFAULT);
    // SOBEL_FILTER_GY 為 y 方向 kernel
}

// Harris 角點偵測：回傳所有角點座標
vector<Point> my_harris_detector(const Mat& src, float threshold = 1e7) {
    // 1. 計算 x, y 方向梯度
    Mat grad_x, grad_y;
    my_sobel_x(src, grad_x);
    my_sobel_y(src, grad_y);

    // 2. 計算 Ix^2, Iy^2, IxIy
    Mat Ix2 = grad_x.mul(grad_x);
    Mat Iy2 = grad_y.mul(grad_y);
    Mat IxIy = grad_x.mul(grad_y);

    // 3. 高斯平滑 (用 3x3 Gaussian_Filter)
    // 讓每個像素的角點響應值 R 能反映周圍像素的梯度分布

    Mat Sx2, Sy2, Sxy;
	my_blur(Ix2, Sx2);
	my_blur(Iy2, Sy2);
	my_blur(IxIy, Sxy);


    // 4. 計算 Harris R 值
    vector<Point> keypoints;
    double k = 0.04;
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            float a = Sx2.at<float>(y, x);
            float b = Sxy.at<float>(y, x);
            float c = Sy2.at<float>(y, x);

            float detM = a * c - b * b;
            float traceM = a + c;
            float R = detM - k * traceM * traceM;

            if (R > threshold) {
                keypoints.push_back(Point(x, y));
            }
        }
    }
    return keypoints;
}

/*==================== SIFT ================================*/
/* SIFT解釋 
用途: 影像特徵點偵測與描述
流程: 
Dog (差分高斯) 金字塔構建
1. 對影像進行高斯模糊
2. 計算影像梯度 (Sobel)
3. 計算每個像素的梯度幅值與方向
4. 對每個關鍵點周圍的區域進行描述子計算
輸出: 關鍵點位置與描述子向量
*/
class descriptor {
public:
    Point point;
    vector<float> orientation_vector;
};


float get_magnitude(Mat image_gray_blur_pad, Mat grad_x, Mat grad_y, Point point) {
    /*
    * input: image and coordinate x y
    * algorithm:
    *         squrt((L(x+1,y) - L(x-1,y))^2 + (L(x, y+1) - L(x, y-1))^2)
    * output: m(x,y)
    */

    float x_distance = image_gray_blur_pad.at<float>(point.y, point.x + 1) - image_gray_blur_pad.at<float>(point.y, point.x - 1);
    float y_distance = image_gray_blur_pad.at<float>(point.y + 1, point.x) - image_gray_blur_pad.at<float>(point.y - 1, point.x);
    return sqrt(x_distance * x_distance + y_distance * y_distance);
}

int get_orientation(const Mat& image_gray_blur_pad, const Mat& grad_x, const Mat& grad_y, const Point& point){
    // 以 float 型別存取，中心差分計算梯度
    float x_distance = image_gray_blur_pad.at<float>(point.y, point.x + 1) - image_gray_blur_pad.at<float>(point.y, point.x - 1);
    float y_distance = image_gray_blur_pad.at<float>(point.y + 1, point.x) - image_gray_blur_pad.at<float>(point.y - 1, point.x);

    // 計算梯度方向（弧度）
    float angle = atan2(y_distance, x_distance);
    if (angle < 0) angle += 2 * PI; // 正規化到 [0, 2PI)

    // 找到對應的方向 bin（0~7）
    int orientation = 0;
    for (int i = 0; i < 8; ++i) {
        if (ORIENTATION[i] <= angle && angle < ORIENTATION[i + 1]) {
            orientation = i;
            break;
        }
    }

    return orientation;
}
vector<float> get_orientation_vector(const Mat& image_gray_blur_pad, const Mat& grad_x_pad, const Mat& grad_y_pad, const Point& point, float weight){

    float magnitude = get_magnitude(image_gray_blur_pad, grad_x_pad, grad_y_pad, point);
    int orientation = static_cast<int>(get_orientation(image_gray_blur_pad, grad_x_pad, grad_y_pad, point));
    vector<float> orientation_vector(8, 0.0f);

    if (orientation >= 0 && orientation < 8) {
        orientation_vector[orientation] += magnitude * weight;
    }

    return orientation_vector;
}
vector<float> get_group_orientation_vector(const Mat& image_gray_blur_pad, const Mat& grad_x_pad, const Mat& grad_y_pad, Point left_top_point) {
    // 8 維方向直方圖，初始為 0
    vector<float> group_hist(8, 0.0f);

    for (int i = left_top_point.x; i < left_top_point.x + 4; ++i) {
        for (int j = left_top_point.y; j < left_top_point.y + 4; ++j) {
            Point pt(i, j);
            vector<float> hist = get_orientation_vector(image_gray_blur_pad, grad_x_pad, grad_y_pad, pt, 1.0f);
            // 直接累加
            for (int k = 0; k < 8; ++k) {
                group_hist[k] += hist[k];
            }
        }
    }
    return group_hist;
}

descriptor get_descriptor(const Mat& image_gray_blur_pad, const Mat& grad_x_pad, const Mat& grad_y_pad, Point keypoint) {
    // 將 keypoint 平移到 padding 後的位置
    Point center = keypoint + Point(9, 9);
    descriptor output;
    output.point = keypoint; // 保留原始座標

    // 16x16 區域分成 4x4 小區塊，每塊 4x4
	// 每個小區塊計算 8 維向量
	// 最後組合成 128 維向量的描述子(descriptor)
    for (int dx = -8; dx < 8; dx += 4) {
        for (int dy = -8; dy < 8; dy += 4) {
            Point block_topleft = center + Point(dx, dy);
            vector<float> hist = get_group_orientation_vector(image_gray_blur_pad, grad_x_pad, grad_y_pad, block_topleft);
            output.orientation_vector.insert(output.orientation_vector.end(), hist.begin(), hist.end());
        }
    }

    // L2 normalization + 截斷
    float norm = 0.0f;
    for (float v : output.orientation_vector) norm += v * v;
    norm = sqrt(norm);
    if (norm > 0) {
        for (float& v : output.orientation_vector) {
            v /= norm;
            if (v > 0.2f) v = 0.2f;
        }
        // 再次 L2 normalization（SIFT 標準流程）
        norm = 0.0f;
        for (float v : output.orientation_vector) norm += v * v;
        norm = sqrt(norm);
        if (norm > 0) for (float& v : output.orientation_vector) v /= norm;
    }

    // 主方向循環平移
    auto max_it = max_element(output.orientation_vector.begin(), output.orientation_vector.end());
    int max_idx = static_cast<int>(distance(output.orientation_vector.begin(), max_it));
    if (max_idx != 0) {
        rotate(output.orientation_vector.begin(),
            output.orientation_vector.begin() + max_idx,
            output.orientation_vector.end());
    }

    return output;
}

vector<descriptor> get_all_descipter(Mat image_gray_blur, Mat grad_x, Mat grad_y, vector<Point> keypoints) {
    /*
    * input:
        - image, keypoints
    * algorithm:
    *   - padding the image
    *   - find all keypoints descriptor
    * output:
        - [[Point, orientation_]...[...]] (N,2,(2, 128))
    */
    vector<descriptor> output;
    //Mat image_gray_blur_pad;
    Mat grad_x_pad, grad_y_pad, image_gray_blur_pad;

    // 做 padding
	// 這裡的 padding 是為了讓每個 keypoint 都能在 16x16 的區域內
    copyMakeBorder(grad_x, grad_x_pad, 9, 9, 9, 9, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(grad_y, grad_y_pad, 9, 9, 9, 9, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(image_gray_blur, image_gray_blur_pad, 9, 9, 9, 9, BORDER_CONSTANT, Scalar(0));
    //

    for (Point keypoint : keypoints) {
        descriptor tmp_descipter;
        tmp_descipter = get_descriptor(image_gray_blur_pad, grad_x_pad, grad_y_pad, keypoint);
        output.push_back(tmp_descipter);
    }


    return output;
}

float cal_dis(descriptor d_1, descriptor d_2) {
    float dis = 0;
    int cnt = 0;
    for (size_t i = 0; i < d_1.orientation_vector.size(); ++i) {
        float diff = d_1.orientation_vector[i] - d_2.orientation_vector[i];
        dis += diff * diff;
        if (d_1.orientation_vector[i] == 0 && d_2.orientation_vector[i] == 0) cnt++;
    }
    dis = sqrt(dis);
    if (cnt > d_1.orientation_vector.size() / 2) dis = 10;
    return dis;
}
/*==================== match keypoints ===================*/
// 匹配點結構
struct MatchPair {
    int idx1;
    int idx2;
};

// 計算單應性矩陣（用 OpenCV findHomography 也可）
Mat computeHomography(const vector<Point2f>& src, const vector<Point2f>& dst) {
    return findHomography(src, dst, 0);
}
// RANSAC 版本的 get_match_keypoints
vector<pair<int, int>> get_match_keypoints(
    const vector<descriptor>& descriptor_image_1,
    const vector<descriptor>& descriptor_image_2,
    int ransac_iter = 10000,
    float match_threshold = 0.25,
    float ransac_inlier_threshold = 2.0f)
{
	//cout << "get_match_keypoints" << endl;

    int max_desc = 5000;
    int n1 = min((int)descriptor_image_1.size(), max_desc);
    int n2 = min((int)descriptor_image_2.size(), max_desc);

    // 1. 初步最近鄰匹配
	cout << n1 << " " << n2 << endl;
    vector<MatchPair> initial_matches;
	int total_comparisons = n1 * n2;
    for (int i = 0; i < n1; ++i) {
        float min_dist = 1e9;
        int min_j = -1;
        for (int j = 0; j < n2; ++j) {
            //cout << "Matching: " << (i * n2 + j + 1) << "/" << total_comparisons << "\r" << flush;
            float dist = cal_dis(descriptor_image_1[i], descriptor_image_2[j]);
            if (dist < min_dist) {
                min_dist = dist;
                min_j = j;
            }
        }
        if (min_dist < match_threshold) {
            initial_matches.push_back({ i, min_j });
        }
    }
    if (initial_matches.size() < 4) return {}; // 不足4組無法RANSAC

    // 2. RANSAC 主流程
    int best_inlier_count = 0;
    vector<MatchPair> best_inliers;
    random_device rd;
    mt19937 gen(rd());

    for (int iter = 0; iter < ransac_iter; ++iter) {
		cout << "RANSAC iteration: " << iter + 1 << "/" << ransac_iter << "\r" << flush;
        // 隨機選4組配對
        vector<int> indices(initial_matches.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), gen);

        vector<Point2f> src_pts, dst_pts;
        for (int k = 0; k < 4; ++k) {
            int idx = indices[k];
            src_pts.push_back(descriptor_image_1[initial_matches[idx].idx1].point);
            dst_pts.push_back(descriptor_image_2[initial_matches[idx].idx2].point);
        }
        Mat H = computeHomography(src_pts, dst_pts);
        if (H.empty()) continue;

        // 計算所有配對的 inlier 數
        vector<MatchPair> inliers;
        for (const auto& match : initial_matches) {
            Point2f pt1 = descriptor_image_1[match.idx1].point;
            Point2f pt2 = descriptor_image_2[match.idx2].point;
            // 轉換 pt1
            Mat pt1_h = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
            Mat pt2_proj = H * pt1_h;
            if (fabs(pt2_proj.at<double>(2, 0)) < 1e-6) continue;
            pt2_proj /= pt2_proj.at<double>(2, 0);
            float dx = pt2.x - pt2_proj.at<double>(0, 0);
            float dy = pt2.y - pt2_proj.at<double>(1, 0);
            float error = sqrt(dx * dx + dy * dy);
            if (error < ransac_inlier_threshold) {
                inliers.push_back(match);
            }
        }
        if (inliers.size() > best_inlier_count) {
            best_inlier_count = static_cast<int>(inliers.size());
            best_inliers = inliers;
        }
    }

    // 3. 回傳最佳 inlier 配對
    vector<pair<int, int>> output;
    for (const auto& match : best_inliers) {
        output.emplace_back(match.idx1, match.idx2);
    }
    return output;
}

/*=============================== main function =============================*/
int main() {
    /*
    //vector<pair<int, int>> match_kepoints;

    //vector<descriptor> descriptor_image_1, descriptor_image_2;
    // read image
    Mat image_1 = imread("01.JPG", IMREAD_COLOR);
    Mat image_2 = imread("02.JPG", IMREAD_COLOR);

	// convert to gray
    Mat  image_gray_1, image_gray_2;
    cvtColor(image_1, image_gray_1, COLOR_BGR2GRAY);
    cvtColor(image_2, image_gray_2, COLOR_BGR2GRAY);

	// 從 CV_8U 轉成 CV_32F (int -> float)
	image_gray_1.convertTo(image_gray_1, CV_32F);
    image_gray_2.convertTo(image_gray_2, CV_32F);

	// Gaussian blur
    // 在做 Sobel 前，對原始影像做高斯平滑（如 my_blur），目的是抑制雜訊。
    Mat  image_gray_blur_1, image_gray_blur_2;
	my_blur(image_gray_1, image_gray_blur_1);
	my_blur(image_gray_2, image_gray_blur_2);
    imwrite("image_gray_blur_1.jpg", image_gray_blur_1);
    imwrite("image_gray_blur_2.jpg", image_gray_blur_2);

	// Sobel filter to get gradient
	// 找到影像的邊緣
	Mat grad_x_1, grad_y_1, grad_x_2, grad_y_2;
	my_sobel_x(image_gray_blur_1, grad_x_1);
	my_sobel_y(image_gray_blur_1, grad_y_1);
	my_sobel_x(image_gray_blur_2, grad_x_2);
	my_sobel_y(image_gray_blur_2, grad_y_2);
	imwrite("grad_x_1.jpg", grad_x_1);
	imwrite("grad_y_1.jpg", grad_y_1);
	imwrite("grad_x_2.jpg", grad_x_2);
	imwrite("grad_y_2.jpg", grad_y_2);
    
	// Harris corner detection
    vector<Point> keypoints_1, keypoints_2;
    keypoints_1 = my_harris_detector(image_gray_blur_1);
    keypoints_2 = my_harris_detector(image_gray_blur_2);
    // 在 main() 最後加上這段，將 keypoints_1 顯示在黑色影像上（白色點）
 //   Mat black_image_1 = Mat::zeros(image_gray_1.size(), CV_8UC3); // 建立黑色底圖 (3通道)
	//Mat black_image_2 = Mat::zeros(image_gray_2.size(), CV_8UC3); // 建立黑色底圖 (3通道)
 //   for (const Point& pt : keypoints_1) {
 //       circle(black_image_1, pt, 2, Scalar(255, 255, 255), -1); // 白色圓點
 //   }
 //   for (const Point& pt : keypoints_2) {
 //       circle(black_image_2, pt, 2, Scalar(255, 255, 255), -1); // 白色圓點
	//}
 //   imwrite("keypoints_1.jpg", black_image_1);
	//imwrite("keypoints_2.jpg", black_image_2);

	// calculate SIFT descriptor
	vector<descriptor> descriptor_image_1, descriptor_image_2;

    //descriptor_image_1 = get_all_descipter(image_gray_blur_1,grad_x_1, grad_y_1, keypoints_1);
    //descriptor_image_2 = get_all_descipter(image_gray_blur_2,grad_x_2, grad_y_2, keypoints_2);
    // 1. 建立 SIFT 物件
    // 
    // 
    
    // SIFT 檢測與描述
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    Mat img1_8u, img2_8u;
    image_gray_1.convertTo(img1_8u, CV_8U);  // 直接用原圖，不用模糊後的
    image_gray_2.convertTo(img2_8u, CV_8U);

    sift->detectAndCompute(img1_8u, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2_8u, noArray(), keypoints2, descriptors2);

    cout << "找到特徵點：" << keypoints1.size() << " / " << keypoints2.size() << endl;

    // 特徵匹配
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Ratio test
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); ++i) {
        if (knn_matches[i].size() >= 2 &&
            knn_matches[i][0].distance < 0.2f * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cout << "找到 " << good_matches.size() << " 組配對點" << endl;
    if (good_matches.size() < 4) {
        cout << "配對點不足，無法計算 Homography！" << endl;
        return -1;
    }

    // 畫出匹配結果
    Mat img_matches;
    drawMatches(image_1, keypoints1, image_2, keypoints2, good_matches, img_matches,
        Scalar::all(-1), Scalar::all(-1), vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite("matches.jpg", img_matches);

    // 取得配對點
    vector<Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(keypoints1[m.queryIdx].pt);
        pts2.push_back(keypoints2[m.trainIdx].pt);
    }

    // 計算 Homography
    cout << "計算 Homography..." << endl;
    Mat H = findHomography(pts2 , pts1, RANSAC, 1.0);
    if (H.empty()) {
        cout << "Homography 計算失敗！" << endl;
        return -1;
    }
    cout << "Homography matrix:\n" << H << endl;

    // 影像拼接
   // 1. 先建立以 image1 為底的結果圖
    Mat result = Mat::zeros(Size(image_1.cols * 2, image_1.rows * 1.2), image_1.type());
    image_1.copyTo(result(Rect(0, 0, image_1.cols, image_1.rows)));

    // 4. 變換 image2 並疊加到 result 上
    Mat warped_image2;
    warpPerspective(image_2, warped_image2, H, result.size());

    // 5. 混合兩張圖
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            Vec3b pixel1 = result.at<Vec3b>(y, x);
            Vec3b pixel2 = warped_image2.at<Vec3b>(y, x);

            // 如果兩個位置都有像素，取平均值
            if (pixel1 != Vec3b(0, 0, 0) && pixel2 != Vec3b(0, 0, 0)) {
                Vec3b blended;
                for (int c = 0; c < 3; c++)
                    blended[c] = uchar((int(pixel1[c]) + int(pixel2[c])) / 2);
                result.at<Vec3b>(y, x) = blended;
            }
            // 如果只有 image2 有像素，則使用 image2 的值
            else if (pixel2 != Vec3b(0, 0, 0)) {
                result.at<Vec3b>(y, x) = pixel2;
            }
            // 如果只有 image1 有像素，保持原值
        }
    }

    // 6. 裁切黑邊
    Mat gray;
    cvtColor(result, gray, COLOR_BGR2GRAY);
    Mat mask = gray > 0;
    Rect bbox = boundingRect(mask);
    Mat cropped = result(bbox);

    // 7. 儲存結果
    imwrite("panorama_image1_base1.jpg", result);
    imwrite("panorama_image1_base_cropped1.jpg", cropped);
   
    //
    */
    // 1. 偵測 ORB 特徵點
    Mat img1 = imread("01.JPG");
    Mat img2 = imread("02.JPG");

    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    orb->detectAndCompute(img1, noArray(), kp1, desc1);
    orb->detectAndCompute(img2, noArray(), kp2, desc2);

    // 2. 特徵點匹配
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // 3. 選擇最佳匹配（簡單方法）
    sort(matches.begin(), matches.end());
    matches.resize(matches.size() * 0.15); // 取前15%好點

    // 4. 計算 Homography
    vector<Point2f> pts1, pts2;
    for (auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
    Mat H = findHomography(pts2, pts1, RANSAC);

    // 5. 拼接影像
    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, img1.rows));
    img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));

    imwrite("Stitched.jpg", result);
    waitKey(0);
    return 0;

    return 0;
    
}