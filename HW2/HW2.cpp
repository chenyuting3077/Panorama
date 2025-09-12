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
vector<Point> my_harris_detector(const Mat& src, float threshold = 1e6) {
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
    // 直接使用預計算的梯度值
    float x_grad = grad_x.at<float>(point.y, point.x);
    float y_grad = grad_y.at<float>(point.y, point.x);
    return sqrt(x_grad * x_grad + y_grad * y_grad);
}


int get_orientation(const Mat& image_gray_blur_pad, const Mat& grad_x, const Mat& grad_y, const Point& point) {
    float x_grad = grad_x.at<float>(point.y, point.x);
    float y_grad = grad_y.at<float>(point.y, point.x);

    float angle = atan2(y_grad, x_grad);
    if (angle < 0) angle += 2 * PI;

    // 將角度量化到 8 個 bin（每個 π/4）
    int orientation = static_cast<int>((angle * 8) / (2 * PI)) % 8;

    return orientation;
}

vector<float> get_orientation_vector(const Mat& image_gray_blur_pad, const Mat& grad_x_pad, const Mat& grad_y_pad, const Point& point, float weight) {
    float magnitude = get_magnitude(image_gray_blur_pad, grad_x_pad, grad_y_pad, point);
    int orientation = get_orientation(image_gray_blur_pad, grad_x_pad, grad_y_pad, point);

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
    Point center = keypoint + Point(9, 9);
    descriptor output;
    output.point = keypoint;

    // 16x16 區域分成 4x4 小區塊
    for (int dx = -8; dx < 8; dx += 4) {
        for (int dy = -8; dy < 8; dy += 4) {
            Point block_topleft = center + Point(dx, dy);
            vector<float> hist = get_group_orientation_vector(image_gray_blur_pad, grad_x_pad, grad_y_pad, block_topleft);
            output.orientation_vector.insert(output.orientation_vector.end(), hist.begin(), hist.end());
        }
    }

    // L2 normalization + 截斷 + 再次正規化
    float norm = 0.0f;
    for (float v : output.orientation_vector) norm += v * v;
    norm = sqrt(norm);

    if (norm > 0) {
        for (float& v : output.orientation_vector) {
            v /= norm;
            if (v > 0.2f) v = 0.2f;
        }

        // 再次正規化
        norm = 0.0f;
        for (float v : output.orientation_vector) norm += v * v;
        norm = sqrt(norm);
        if (norm > 0) {
            for (float& v : output.orientation_vector) v /= norm;
        }
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
    if (d_1.orientation_vector.size() != d_2.orientation_vector.size()) {
        return 1e9; // 維度不匹配
    }

    float dis = 0;
    for (size_t i = 0; i < d_1.orientation_vector.size(); ++i) {
        float diff = d_1.orientation_vector[i] - d_2.orientation_vector[i];
        dis += diff * diff;
    }

    return sqrt(dis); // 標準 L2 距離
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
    int ransac_iter = 20000,
    float match_threshold = 0.3,
    float ransac_inlier_threshold = 2.0f)
{
	//cout << "get_match_keypoints" << endl;
    cout << descriptor_image_1.size() << " " << descriptor_image_2.size() << endl;
    int max_desc = 10000;
    int n1 = min((int)descriptor_image_1.size(), max_desc);
    int n2 = min((int)descriptor_image_2.size(), max_desc);

    // 1. 初步最近鄰匹配
	cout << n1 << " " << n2 << endl;
    vector<MatchPair> initial_matches;
    float ratio_threshold = 0.7f;
    // 改用 Ratio Test 進行初步匹配
    for (int i = 0; i < n1; ++i) {
        float min_dist = 1e9;
        float second_min_dist = 1e9;
        int min_j = -1;

        for (int j = 0; j < n2; ++j) {
            float dist = cal_dis(descriptor_image_1[i], descriptor_image_2[j]);
            if (dist < min_dist) {
                second_min_dist = min_dist;
                min_dist = dist;
                min_j = j;
            }
            else if (dist < second_min_dist) {
                second_min_dist = dist;
            }
        }

        // Ratio Test: 最近距離 / 次近距離 < 閾值
        if (min_j != -1 && second_min_dist > 0 &&
            min_dist / second_min_dist < ratio_threshold) {
            initial_matches.push_back({ i, min_j });
        }
    }

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

            if (best_inlier_count > initial_matches.size() * 0.8) {
                cout << "\n提早終止 RANSAC，找到穩定解" << endl;
                break;
            }
        }


    }

    // 3. 回傳最佳 inlier 配對
    vector<pair<int, int>> output;
    for (const auto& match : best_inliers) {
        output.emplace_back(match.idx1, match.idx2);
    }
    return output;
}

void SIFT() {

    // 1. 建立 SIFT 物件
    // 
    // 

    // SIFT 檢測與描述
    /*
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
            knn_matches[i][0].distance < 0.7f * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cout << "找到 " << good_matches.size() << " 組配對點" << endl;

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
    */
    //Mat H = findHomography(pts2, pts1, RANSAC);

    //// 5. 拼接影像
    //Mat result;
    //warpPerspective(image_2, result, H, Size(image_2.cols + image_2.cols, image_1.rows));
    //image_1.copyTo(result(Rect(0, 0, image_1.cols, image_1.rows)));

    //imwrite("Stitched_SIFT.jpg", result);
}


// 生成與影像大小相同的權重影像
Mat generateWeightImage(const Size& size, bool isLeft) {
    Mat weight(size, CV_32F);

    for (int y = 0; y < size.height ; ++y) {
        for (int x = 0; x < size.width ; ++x) {
            float linearWeight = static_cast<float>(x) / size.width;
            if (isLeft) {
                // 左側影像的權重從 1 遞減到 0，使用平方根調整
                weight.at<float>(y, x) = 1.0f - sqrt(linearWeight);
            }
            else {
                // 右側影像的權重從 0 遞增到 1，使用平方根調整
                weight.at<float>(y, x) = sqrt(linearWeight);
            }
        }
    }
    // 將單通道影像複製到三個通道
    Mat weight3Channel;
    vector<Mat> channels(3, weight); // 複製三份單通道影像
    merge(channels, weight3Channel); // 合併為三通道影像

    return weight3Channel;
}
/*=============================== main function =============================*/
int main() {
    
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
	vector<pair<int, int>> match_kepoints;
    descriptor_image_1 = get_all_descipter(image_gray_blur_1,grad_x_1, grad_y_1, keypoints_1);
    descriptor_image_2 = get_all_descipter(image_gray_blur_2,grad_x_2, grad_y_2, keypoints_2);
    match_kepoints = get_match_keypoints(descriptor_image_1, descriptor_image_2);
	cout << "找到 " << match_kepoints.size() << " 組配對點" << endl;

	// 畫出匹配結果
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> good_matches;

    // 轉換關鍵點格式
    for (size_t i = 0; i < keypoints_1.size(); ++i) {
        keypoints1.push_back(KeyPoint(keypoints_1[i], 1.0f));
    }
    for (size_t i = 0; i < keypoints_2.size(); ++i) {
        keypoints2.push_back(KeyPoint(keypoints_2[i], 1.0f));
    }

    // 轉換匹配結果格式
    for (size_t i = 0; i < match_kepoints.size(); ++i) {
        good_matches.push_back(DMatch(match_kepoints[i].first, match_kepoints[i].second, 0.0f));
    }

    Mat img_matches;
    drawMatches(image_1, keypoints1, image_2, keypoints2, good_matches, img_matches,
        Scalar::all(-1), Scalar::all(-1), vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite("matches.jpg", img_matches);

    vector<Point2f> pts1, pts2;
    for (const auto& match : match_kepoints) {
        pts1.push_back(Point2f(keypoints_1[match.first].x, keypoints_1[match.first].y));
        pts2.push_back(Point2f(keypoints_2[match.second].x, keypoints_2[match.second].y));
    }

    Mat H = findHomography(pts2, pts1, RANSAC);
    Mat result_without_blending;
    warpPerspective(image_2, result_without_blending, H, Size(image_1.cols + image_2.cols, image_1.rows));
    // 將 image_1 複製到 blended 的左側
    image_1.copyTo(result_without_blending(Rect(0, 0, image_1.cols, image_1.rows)));
	imwrite("without_blened_result.jpg", result_without_blending);


    Mat result;
    // 確保影像類型一致
    warpPerspective(image_2, result, H, Size(image_1.cols + image_2.cols, image_1.rows));
    image_1.convertTo(image_1, CV_32FC3);
    result.convertTo(result, CV_32FC3);

    // 建立融合結果影像
    Mat blended = Mat::zeros(result.size(), result.type());
    Mat big_image_1 = Mat::zeros(result.size(), result.type());

    // 將 image_1 複製到 big_image_1 的左側
    int blend_start = max(0, image_1.cols - 500); // 確保範圍合法
    int blend_end = min(result.cols, image_1.cols); // 確保範圍合法

    image_1.copyTo(big_image_1(Rect(0, 0, image_1.cols, image_1.rows)));

    // 線性混合
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            if (x < blend_start) {
                // 左側影像完全保留
                blended.at<Vec3f>(y, x) = big_image_1.at<Vec3f>(y, x);
            }
            else if (x >= blend_start && x < blend_end) {
                // 線性混合區域
                float alpha = static_cast<float>(x - blend_start) / (blend_end - blend_start); // 混合比例
                blended.at<Vec3f>(y, x) = big_image_1.at<Vec3f>(y, x) * (1.0f - alpha) + result.at<Vec3f>(y, x) * alpha;
            }
            else {
                // 右側影像完全保留
                blended.at<Vec3f>(y, x) = result.at<Vec3f>(y, x);
            }
        }
    }

    // 將結果轉回 8 位元格式並保存
    blended.convertTo(blended, CV_8U);
    imwrite("blended_result_linear.jpg", blended);

	//imwrite("Stitched.jpg", result);
    return 0;
}