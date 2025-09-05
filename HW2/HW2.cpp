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
    for (int i = 0;i < 128;i++) {
        //cout << "d_1.orientation_vector["<< i <<"]: " << d_1.orientation_vector[i] << endl;
        //cout << "d_2.orientation_vector["<< i <<"]: " << d_2.orientation_vector[i] << endl;
        dis += pow(d_1.orientation_vector[i] - d_2.orientation_vector[i], 2);
        
        if (d_1.orientation_vector[i] == 0 && d_2.orientation_vector[i]==0){
            cnt++;
        }     
    }
    //cout << dis << endl;
    /*if (cnt == 0) {

        cout << "descriptor_1: ";
        for (int i = 0;i < d_1.orientation_vector.size();i++) {
            cout << d_1.orientation_vector[i] << " ";
        }
        cout << endl;
        cout << "descriptor_2: " << endl;
        for (int i = 0;i < d_2.orientation_vector.size();i++) {
            cout << d_2.orientation_vector[i] << " ";
        }
        cout << endl;

    }*/
    //if (dis != 0) {
       // cout << dis << endl;
   // }
    
    dis = sqrt(dis);
    if (cnt > 64) {
        dis = 10;
    }
    //cout << "dis: " << dis << endl;
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
    float match_threshold = 0.5,
    float ransac_inlier_threshold = 3.0f)
{
	cout << "get_match_keypoints" << endl;

    int max_desc = 300;
    int n1 = min((int)descriptor_image_1.size(), max_desc);
    int n2 = min((int)descriptor_image_2.size(), max_desc);

    // 1. 初步最近鄰匹配
	cout << descriptor_image_1.size() << " " << descriptor_image_2.size() << endl;
    vector<MatchPair> initial_matches;
    for (int i = 0; i < n1; ++i) {
        float min_dist = 1e9;
        int min_j = -1;
        for (int j = 0; j < n2; ++j) {
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



/*==================== evaluate_coorspondences ===================*/
int evaluate_coorspondences() {

}
/*==================== transform===================*/
Mat transform(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector< DMatch > good_matches) {
    /*
    input: four source point and four distination point
    output: transformation matrix
    */
}

Mat find_homography_(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2){
    /*
    * input: four matched point
    * output: homography matrix
    */
    //cout << "find_homography_" << endl;
    Mat min_eigen_vec;
    Mat2d A;
    for (int i = 0;i < 4;i++) {
        Point source = keypoints1[i].pt;
        Point destination = keypoints2[i].pt;

        //cout << source << " " << destination << endl;
        Mat tmp =
            (Mat_<double>(2, 9) <<
                source.x, source.y, 1, 0, 0, 0, -destination.x * source.x, -destination.x * source.y, -destination.x,
                0, 0, 0, source.x, source.y, 1, -destination.y * source.x, -destination.y * source.y, -destination.y);

        //vconcat(output, tmp, output);
        A.push_back(tmp);
    }
    A = A.t() * A;
    //cout << "A.t() * A; " << A.size() << endl;
    Mat U, W, Vt;

   // Vt.convertTo(Vt, CV_64F);
    //U.convertTo(U, CV_64F);
    //W.convertTo(W, CV_64F);
    //min_eigen_vec.convertTo(min_eigen_vec, CV_64F);
    //做 SVD 分解
    SVD::compute(A, W, U, Vt);
    min_eigen_vec = Vt.row(Vt.rows - 1);
    min_eigen_vec = min_eigen_vec * (1 / min_eigen_vec.at<double>(0, 8));
    min_eigen_vec = min_eigen_vec.reshape(0, 3);

    //cout << min_eigen_vec << endl;
    return min_eigen_vec;
}


Mat transform_test(Mat image_1, Mat image_2, vector<KeyPoint> keypoints1_all, vector<KeyPoint> keypoints2_all) {
    /*
    input: matched point point and four distination point
    output: transformation matrix
    */
    /*
    vector<KeyPoint> keypoints1_all;
    vector<KeyPoint> keypoints2_all;
    vector<Point> points1_all;
    vector<Point> points2_all;
    vector< DMatch > good_matches;

    Mat image_1 = imread("C:/Users/Allen/Desktop/HW2/01.JPG", CV_8U);
    Mat image_2 = imread("C:/Users/Allen/Desktop/HW2/02.JPG", CV_8U);
   
    ifstream matched("matched.txt");
    string line;
    for(int i=0;i< {
        
        stringstream ss;
        float source_x = 0, source_y=0;
        float des_x = 0, des_y = 0;

        ss << line;
        ss >> source_x >> source_y >> des_x >> des_y;
        KeyPoint source, des;
        Point source_pt(source_x, source_y), des_pt(des_x, des_y);
        source.pt = source_pt;
        des.pt = des_pt;
        keypoints1_all.push_back(source);
        keypoints2_all.push_back(des);
        points1_all.push_back(source_pt);
        points2_all.push_back(des_pt);
    }
    */
    Mat min_eigen_vec, h, min_eigen_vec_inverse;
    /*========================================================================*/
    /*
    * RANSAC
    * input: keypoint list
    * output: homorgrphy
    * 1. random choose the 4 point.
    * 2. compute homorgrphic
    * 3. compute the error
    *
    */
    auto RANSAC_begin = chrono::steady_clock::now();
    
    
    Mat gloabl_homography_inv, gloabl_homography;
    float global_dis = 1e9;
    vector<Point> points1;
    vector<Point> points2;
    for (int i = 0;i < 2000;i++) {
        /*random_get_four_matched_point
        * 
        * */
        vector<KeyPoint> keypoints1;
        vector<KeyPoint> keypoints2;
        vector<Point> points1_tmp;
        vector<Point> points2_tmp;
        /*如果重複就break*/
        vector<int> is_selected;
        for (int j = 0;j < 4;j++) {
            
            
            float dis_x, dis_y, dis;
            bool is_random__near = true;
            int index = -1;
            //do not get so near & do tno repeat//
            while (is_random__near) {
                index = rand() % (keypoints1_all.size());

                //太近
                if (keypoints1.size() > 0) {
                    for (int i = 0;i < keypoints1.size();i++) {
                        //重複
                        if ((keypoints1_all[index].pt.x == keypoints1[i].pt.x )|| 
                            (keypoints1_all[index].pt.y == keypoints1[i].pt.y )||
                            (keypoints2_all[index].pt.x == keypoints2[i].pt.x) ||
                            (keypoints2_all[index].pt.y == keypoints2[i].pt.y)
                            ) {
                            ///cout << "is_repeat" << endl;
                            is_random__near = true;
                            break;
                        }else {

                            dis_x = keypoints1_all[index].pt.x - keypoints1_all[i].pt.x;
                            dis_y = keypoints1_all[index].pt.y - keypoints1_all[i].pt.y;

                            dis = sqrt(pow(dis_x, 2) + pow(dis_y, 2));
                            
                            if (dis > 10) {
                                is_random__near = false;
                            } 
                            else{
                                is_random__near = true;
                                break;
                            }
                        }

                    }
                }
                else {
                    is_random__near = false;
                }
            }
            is_selected.push_back(index);
            keypoints1.push_back(keypoints1_all[index]);
            keypoints2.push_back(keypoints2_all[index]);

            points1_tmp.push_back(keypoints1_all[index].pt);
            points2_tmp.push_back(keypoints2_all[index].pt);
        }

        min_eigen_vec = find_homography_(keypoints1, keypoints2);
        //cout << min_eigen_vec_inverse << endl;
        //計算誤差
        float local_dis = 0;
        for (int j = 0;j < keypoints1_all.size();j++) {
            Point point = keypoints1_all[j].pt;
            Mat new_point = (Mat_<double>(3, 1) << point.x, point.y, 1);
            Mat old_point = min_eigen_vec * new_point;
            old_point = old_point / old_point.at<double>(0, 2);
            float old_x = (int)old_point.at<double>(0, 0);
            float old_y = (int)old_point.at<double>(0, 1);
            local_dis += sqrt(pow((keypoints2_all[j].pt.y - old_y),2) + pow(keypoints2_all[j].pt.x - old_x,2));
        }
        //cout << "local_dis " << local_dis <<  "global_dis " << global_dis << endl;
        if (local_dis < global_dis) {
            
            points1 = points1_tmp;
            points2 = points2_tmp;
            gloabl_homography_inv = min_eigen_vec.inv();
            gloabl_homography = min_eigen_vec;
            global_dis = local_dis;
        }
    }
    Mat new_image(image_1.rows, image_1.cols + image_2.cols, CV_8U);
    image_2.copyTo(new_image);

    
    auto RANSAC_end = chrono::steady_clock::now();
    auto RANSAC_elapsed = chrono::duration<double>(RANSAC_end - RANSAC_begin);
    cout << "RANSAC: " << RANSAC_elapsed.count() << " seconds" << endl;
    /*======================================================================*/
    /*inverse mapping*/
    auto Inverse_begin = chrono::steady_clock::now();

    for (int i = 0;i < new_image.rows;i++) {
        for (int j = 0;j < new_image.cols;j++) {
            Point point(j, i);
            Mat new_point = (Mat_<double>(3, 1) << point.x, point.y, 1);
            Mat old_point = gloabl_homography_inv * new_point;
            //Mat old_point = h_inv * new_point;
            old_point = old_point / old_point.at<double>(0, 2);
            float old_x = (int)old_point.at<double>(0, 0);
            float old_y = (int)old_point.at<double>(0, 1);
            if (old_x > 0 && old_y > 0 && old_x < image_1.cols && old_y < image_1.rows) {
                if (new_image.at<uchar>(point.y, point.x) != 0) {
                    new_image.at<uchar>(point.y, point.x) = (image_1.at<uchar>(old_y, old_x) + new_image.at<uchar>(point.y, point.x)) /2 ;
                }
                else {
                    new_image.at<uchar>(point.y, point.x) = image_1.at<uchar>(old_y, old_x);
                }
                
            }
            
        }
    }

    auto Inverse_end = chrono::steady_clock::now();
    auto Inverse_elapsed = chrono::duration<double>(Inverse_end - Inverse_begin);

    cout << "Inverse: " << Inverse_elapsed.count() << " seconds" << endl;
    /*======================================================================*/
    return new_image;
}
/*====================  blend ===================*/
int blend() {

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

    descriptor_image_1 = get_all_descipter(image_gray_blur_1,grad_x_1, grad_y_1, keypoints_1);
    descriptor_image_2 = get_all_descipter(image_gray_blur_2,grad_x_2, grad_y_2, keypoints_2);

	//vector<pair<int, int>> match_kepoints;
 //   match_kepoints = get_match_keypoints(descriptor_image_1, descriptor_image_2);

 //   // 1. 轉成 KeyPoint 格式
 //   vector<KeyPoint> keypoints1, keypoints2;
 //   for (const auto& pt : keypoints_1) keypoints1.push_back(KeyPoint(pt, 1.f));
 //   for (const auto& pt : keypoints_2) keypoints2.push_back(KeyPoint(pt, 1.f));

 //   // 2. 轉成 DMatch 格式
 //   vector<DMatch> good_matches;
 //   for (const auto& match : match_kepoints) {
 //       good_matches.push_back(DMatch(match.first, match.second, 0));
 //       cout << "Matched KeyPoint Image 1 Index: " << match.first 
	//		<< " with Image 2 Index: " << match.second << endl;
 //   }

 //   
 //   // 3. 繪製匹配結果
 //   Mat img_matches;
 //   drawMatches(image_1, keypoints1, image_2, keypoints2, good_matches, img_matches,
 //       Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


 //   //imshow("matches", img_matches);
 //   //imwrite("matches.jpg", img_matches);
 //   //waitKey(0);

 //   // 1. 從 good_matches 取得配對點
 //   vector<Point2f> pts1, pts2;
 //   for (const auto& m : good_matches) {
 //       pts1.push_back(keypoints1[m.queryIdx].pt);
 //       pts2.push_back(keypoints2[m.trainIdx].pt);
 //   }

 //   // 2. 用 RANSAC 找單應性矩陣
 //   Mat H = findHomography(pts1, pts2, RANSAC);

 //   // 3. 影像拼接
 //   Mat result;
 //   warpPerspective(image_1, result, H,
 //       Size(image_1.cols + image_2.cols, max(image_1.rows, image_2.rows)));
 //   Mat roi(result, Rect(0, 0, image_2.cols, image_2.rows));
 //   image_2.copyTo(roi);

 //   // 4. 顯示與儲存
 //   Mat result_small;
 //   resize(result, result_small, Size(), 0.33, 0.33);
 //   imshow("stitch", result_small);
 //   imwrite("stitch.jpg", result);
 //   waitKey(0);

    // 假設 descriptor_image_1, descriptor_image_2 都是 vector<descriptor>
    int n1 = descriptor_image_1.size();
    int n2 = descriptor_image_2.size();


    Mat desc1_mat(n1, 128, CV_32F);
    Mat desc2_mat(n2, 128, CV_32F);
    for (int i = 0; i < n1; ++i)
        memcpy(desc1_mat.ptr<float>(i), descriptor_image_1[i].orientation_vector.data(), 128 * sizeof(float));
    for (int i = 0; i < n2; ++i)
        memcpy(desc2_mat.ptr<float>(i), descriptor_image_2[i].orientation_vector.data(), 128 * sizeof(float));


    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(desc1_mat, desc2_mat, matches);

    // 可選：根據距離過濾
    double min_dist = 1e9, max_dist = 0;
    for (const auto& m : matches) {
        min_dist = min(min_dist, (double)m.distance);
        max_dist = max(max_dist, (double)m.distance);
    }
    vector<DMatch> good_matches;
    for (const auto& m : matches) {
        if (m.distance <= max(2 * min_dist, 0.3)) { // 0.3 可依實際情況調整
            good_matches.push_back(m);
        }
    }


    // 轉成 KeyPoint 格式
    vector<KeyPoint> keypoints1, keypoints2;
    for (const auto& d : descriptor_image_1) keypoints1.push_back(KeyPoint(d.point, 1.f));
    for (const auto& d : descriptor_image_2) keypoints2.push_back(KeyPoint(d.point, 1.f));

    Mat img_matches;
    drawMatches(image_1, keypoints1, image_2, keypoints2, good_matches, img_matches,
        Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("matches", WINDOW_NORMAL); // 允許調整視窗大小
    resizeWindow("matches", 800, 600);    // 設定視窗大小為 800x600
    imshow("matches", img_matches);
    imwrite("matches.jpg", img_matches);
    waitKey(0);


    vector<Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(keypoints1[m.queryIdx].pt);
        pts2.push_back(keypoints2[m.trainIdx].pt);
    }
    if (pts1.size() >= 4 && pts2.size() >= 4) {
        Mat H = findHomography(pts1, pts2, RANSAC);
        if (!H.empty() && H.rows == 3 && H.cols == 3) {
            Mat result;
            warpPerspective(image_1, result, H, Size(image_1.cols + image_2.cols, max(image_1.rows, image_2.rows)));
            Mat roi(result, Rect(0, 0, image_2.cols, image_2.rows));
            image_2.copyTo(roi);
            //imshow("stitch", result);
			
            // 1. 轉成灰階（如果是彩色圖）
            Mat gray;
            if (result.channels() == 3)
                cvtColor(result, gray, COLOR_BGR2GRAY);
            else
                gray = result;

            // 2. 產生二值遮罩（非黑為 255，黑為 0）
            Mat mask;
            threshold(gray, mask, 1, 255, THRESH_BINARY);

            // 3. 找出所有非黑像素座標
            vector<Point> points;
            findNonZero(mask, points);

            // 4. 計算最小包圍盒
            Rect bbox = boundingRect(points);

            // 5. 裁切
            Mat cropped = result(bbox);

            // 6. 儲存
            imwrite("stitch_cropped.jpg", cropped);
            imshow("stitch_cropped", cropped);
            waitKey(0);
        }
        else {
            cout << "Homography matrix H is invalid!" << endl;
        }
    }
    return 0;
    
}