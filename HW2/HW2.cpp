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
using namespace std;
using namespace cv;
using namespace Eigen;
# define PI 3.14159265358979323846
# define ONE_EIGHTH_PI (1.0/4.0*PI)
const float ORIENTATION[9] = { 0, 1*ONE_EIGHTH_PI, 2* ONE_EIGHTH_PI , 3 * ONE_EIGHTH_PI , 4 * ONE_EIGHTH_PI , 5 * ONE_EIGHTH_PI , 6 * ONE_EIGHTH_PI , 7 * ONE_EIGHTH_PI , 8 * ONE_EIGHTH_PI };
const Mat Gaussian_Filter = (Mat_<float>(3, 3) <<
    1, 2, 1,
    2, 4, 2,
    1, 2, 1) / 16.0f;

/*====================  my_blur ===================*/
void my_blur(const Mat& src, Mat& dst) {
    // 在做 Sobel 前，對原始影像做高斯平滑（如 my_blur），目的是抑制雜訊。
    // Point(-1, -1) 表示 kernel 以中心點為 anchor（自動置中）
    // 0 表示不加偏移
    // BORDER_DEFAULT 表示邊界外的像素值由 OpenCV 自動決定
    filter2D(src, dst, -1, Gaussian_Filter, Point(-1, -1), 0, BORDER_DEFAULT);
    // 若要回傳原型態（如 CV_8U），可再轉回
}


class descriptor {
public:
    Point point;
    vector<float> orientation_vector;
};

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
/*====================descriptor================================*/


float get_magnitude(Mat image_gray_blur_pad, Mat grad_x, Mat grad_y, Point point) {
    /*
    * input: image and coordinate x y
    * algorithm:
    *         squrt((L(x+1,y) - L(x-1,y))^2 + (L(x, y+1) - L(x, y-1))^2)
    * output: m(x,y)
    */
    //cout << "get_magnitude" << point << endl;


    float x_distance = (float)image_gray_blur_pad.at<uchar>(point.y, point.x+1) - (float)image_gray_blur_pad.at<uchar>(point.y, point.x - 1);
    float y_distance = (float)image_gray_blur_pad.at<uchar>(point.y + 1, point.x) - (float)image_gray_blur_pad.at<uchar>(point.y - 1, point.x);
    //float x_distance = (float)grad_x.at<uchar>(point.y, point.x);
    //float y_distance = (float)grad_y.at<uchar>(point.y, point.x);

    return  sqrt(pow(x_distance, 2) + pow(y_distance, 2));
}
float get_orientation(Mat image_gray_blur_pad, Mat grad_x, Mat grad_y, Point point) {
    /*
    * input: image and coordinate x y
    * algorithm:
    *         atan2((L(x+1,y) - L(x-1,y))^2 + (L(x, y+1) - L(x, y-1))^2)
    * output: \theta (x,y)
    */
    //cout << "image_gray_blur_pad.at<float>(point.y, point.x + 1 )" << (float)image_gray_blur_pad.at<uchar>(point.y, point.x + 1 ) << endl;
    //cout << "image_gray_blur_pad.at<float>(point.y, point.x - 1)" << (float)image_gray_blur_pad.at<uchar>(point.y, point.x - 1) << endl;
    float x_distance = (float)image_gray_blur_pad.at<uchar>(point.y, point.x + 1) - (float)image_gray_blur_pad.at<uchar>(point.y, point.x - 1);
    float y_distance = (float)image_gray_blur_pad.at<uchar>(point.y + 1, point.x) - (float)image_gray_blur_pad.at<uchar>(point.y - 1, point.x);
    //float x_distance = (float)grad_x.at<uchar>(point.y, point.x);
    //float y_distance = (float)grad_y.at<uchar>(point.y, point.x);
    float radius = atan2(y_distance, x_distance);
    float orientation = -1;

    //cout << "x_distance: " << x_distance << "y_distance " << y_distance << endl;
    for (int i = 0;i < 9;i++) {
        if (radius < 0) {
            radius += 2 * PI;
        }
        if (ORIENTATION[i] <= radius && radius < ORIENTATION[i + 1]) {
            orientation = i;
            break;
        }
        if (i == 8) {
            cout << radius << endl;
            exit(1);
        }
    }
   // namedWindow("grad_x", WINDOW_AUTOSIZE);
   // namedWindow("grad_y", WINDOW_AUTOSIZE);
    //imshow("Display_1", grad_x);
    //imshow("Display_2", grad_y);
    //waitKey(0);
   // destroyAllWindows();
    //cout << "orientation: " << orientation << endl;
    return orientation;
}

vector<float> add_orientation_vector(vector<float> p1, vector<float> p2) {
    //cout << "add_orientation_vector" << endl;
    vector<float> output(8,0);
    for (int i = 0;i < 8;i++) {
        output[i] = p1[i] + p2[i];
    }
    
    return output;
}
vector<float> get_orientation_vector(Mat image_gray_blur_pad, Mat grad_x_pad, Mat grad_y_pads, Point point, float weight) {
    //cout << "get_orientation_vector" << endl;
    float magnitude = get_magnitude(image_gray_blur_pad, grad_x_pad, grad_y_pads, point);
    //float magnitude = 1;
    float orientation = get_orientation(image_gray_blur_pad, grad_x_pad, grad_y_pads, point);
    vector<float> orientation_vector(8, 0);
    orientation_vector[orientation] = orientation_vector[orientation] + magnitude * weight;
    //print_vecotr(orientation_vector, "orientation_vector");
    return orientation_vector;
}
vector<float> get_group_orientation_vector(Mat image_gray_blur_pad, Mat grad_x_pad, Mat grad_y_pad, Point left_top_point) {
    /*
    * input:
        - image, keypoints
    * algorithm:
    *   - padding the image
    *   - find all keypoints descriptor
    * output:
        - [[Point, orientation_]...[...]] (N,2,(2, 128))
    */
    vector<float> group_orientation_vector(8, 0), orientation_vector;

    for (int i = left_top_point.x;i < left_top_point.x + 4;i++) {
        for (int j = left_top_point.y;j < left_top_point.y + 4;j++) {
            //cout << i << " " << j << endl;
            Point point(i, j);
            //cout << point << endl;
            //float offset_x = (float)i - (float)(left_top_point.x + left_top_point.x + 4) / 2;
            //float offset_y = (float)j - (float)(left_top_point.y + left_top_point.y + 4) / 2;
            //float scale_factor = 1.5;
            //float scale = scale_factor * 2 / pow(2, (1));
            //float weight_factor = -0.5 / (scale * scale);
            //float weight = exp(weight_factor * (offset_x * offset_x + offset_y * offset_y));
            orientation_vector = get_orientation_vector(image_gray_blur_pad, grad_x_pad, grad_y_pad, point, 1);
            group_orientation_vector = add_orientation_vector(group_orientation_vector, orientation_vector);
            //print_vecotr(orientation_vector, "orientation_vector");
        }
    }
    //print_vecotr(group_orientation_vector, "group_orientation_vector");
    return group_orientation_vector;
}
descriptor get_descriptor(Mat image_gray_blur_pad, Mat grad_x_pad, Mat grad_y_pad, Point keypoint) {
    //cout << "get_descriptor" << endl;
    /*
    * input:
        - keypoint
    * algorithm:
    *   - traverse 16x16 pixel (keypoint is center)
    *   - each step set as 2
    * output:
        - descipter (this point, orientation vector)
    */
    keypoint.x = keypoint.x + 9;
    keypoint.y = keypoint.y + 9;
    descriptor output;
    for (int bias_x = keypoint.x - 8; bias_x < keypoint.x + 8; bias_x += 4) {
        for (int bias_y = keypoint.y - 8; bias_y < keypoint.y + 8; bias_y += 4) {
            //cout << bias_x << " " << bias_y << endl;
            vector<float> group_orientation_vector;
            Point point(bias_x, bias_y);
            group_orientation_vector = get_group_orientation_vector(image_gray_blur_pad, grad_x_pad, grad_y_pad, point);
            output.orientation_vector.insert(output.orientation_vector.end(), group_orientation_vector.begin(), group_orientation_vector.end());
            //cout << "output.orientation_vector.size(): " << output.orientation_vector.size() << endl;
        }
    }
    int max_element_index = max_element(output.orientation_vector.begin(), output.orientation_vector.end()) - output.orientation_vector.begin();
    float _max_element = *max_element(output.orientation_vector.begin(), output.orientation_vector.end());
    // Normalize
    
    if (_max_element != 0) {
        float L2_sum = 0;
        float threshold = 0.2;
        for (int i = 0;i < output.orientation_vector.size();i++) {
            //output.orientation_vector[i] = output.orientation_vector[i] / _max_element;
            L2_sum += output.orientation_vector[i] * output.orientation_vector[i];
            
            // output.orientation_vector[i] = round(output.orientation_vector[i] * 100) / 100;
        }
        L2_sum = sqrt(L2_sum);
        //cout << "L2_sum: " << L2_sum << endl;
        for (int i = 0;i < output.orientation_vector.size();i++) {
            output.orientation_vector[i] = output.orientation_vector[i] / L2_sum;
            if (output.orientation_vector[i] > threshold) {
               output.orientation_vector[i] = threshold;
            }
        }
        
    }
    
    // rotate
    //print_vecotr(output.orientation_vector, "bEFORE output.orientation_vector: ");
    if (max_element_index != 0) {
        //rotate
        rotate(output.orientation_vector.begin(),
            output.orientation_vector.begin() + max_element_index, // this will be the new first element
            output.orientation_vector.end());
    }
    
    //error detect
    /*
    for (int i = 0;i < output.orientation_vector.size();i++) {
        if (output.orientation_vector[i] > 1 || output.orientation_vector[i]<0) {
            cout << output.orientation_vector[i] << endl;
            exit(2);
        }
         
    }
    */
    //print_vecotr(output.orientation_vector, "AFTER output.orientation_vector: ");
    keypoint.x = keypoint.x - 9;
    keypoint.y = keypoint.y - 9;
    output.point = keypoint;
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
    //cout << "get all descripter" << endl;
    // 做 padding
    copyMakeBorder(grad_x, grad_x_pad, 9, 9, 9, 9, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(grad_y, grad_y_pad, 9, 9, 9, 9, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(image_gray_blur, image_gray_blur_pad, 9, 9, 9, 9, BORDER_CONSTANT, Scalar(0));
    for (Point keypoint : keypoints) {
        descriptor tmp_descipter;
        tmp_descipter = get_descriptor(image_gray_blur_pad, grad_x_pad, grad_y_pad, keypoint);
        output.push_back(tmp_descipter);
    }
    //print_vecotr(tmp_descipter, "tmp_descipter");
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
vector<pair<int, int>> get_match_keypoints(vector<descriptor> descriptor_image_1, vector<descriptor>descriptor_image_2) {
    /*
    * input: image, point
    * algorithm:
    *   - calculate the 16x16 from keypoint. Divide it into four block each block have 4x4 pixels. Each block need to do:
    *       - break a 4x4 block into 2x2 group, each group need to do:
    *           - add up the magnitude according to the magnitude
    * output: descriptor (128 dimensions vector)
    */
    vector<pair<int, int>> output;
    int count = 0;
    for (int i = 0; i < descriptor_image_1.size();i++) {
        descriptor d_1 = descriptor_image_1[i];
        float global_min = 10000000;
        pair<int, int> match_keypoint;
        for (int j = 0; j < descriptor_image_2.size();j++) {
            count++;
            descriptor d_2 = descriptor_image_2[j];
            bool is_equal = equal(d_1.orientation_vector.begin(), d_1.orientation_vector.end(), d_2.orientation_vector.begin());
            //cout << "d_1: " << d_1.point << " d_2: " << d_2.point << " " << cal_dis(d_1, d_2) << endl;
            float local_min = 0;
            
            local_min = cal_dis(d_1, d_2);

            if (local_min <  global_min  ) {
                global_min = local_min;
                match_keypoint.first = i;
                match_keypoint.second = j;
            }
        }
        if (global_min < 0.25) {
            //cout << "global_min: " << global_min << endl;
            output.push_back(match_keypoint);
        }
        else {
            //cout << "global_min: " << global_min << endl;
        }
        
    }
    //cout << count << endl;
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


    //auto detected_end = chrono::steady_clock::now();
    //auto detected_elapsed = chrono::duration<double>(detected_end - detected_begin);
    //cout << "detected: " << detected_elapsed.count() << " seconds" << endl;

    //auto matched_begin = chrono::steady_clock::now();
    //descriptor_image_1 = get_all_descipter(image_gray_blur_1,grad_x_1, grad_y_1, keypoints_1);
    //descriptor_image_2 = get_all_descipter(image_gray_blur_2,grad_x_2, grad_y_2, keypoints_2);
    //match_kepoints = get_match_keypoints(descriptor_image_1, descriptor_image_2);
    //auto matched_end = chrono::steady_clock::now();
    //auto matched_elapsed = chrono::duration<double>(matched_end - matched_begin);
    //cout << "Matched time: " << matched_elapsed.count() << " seconds" << endl;


    //cout << "matched: " << match_kepoints.size() << endl;
    //vector<KeyPoint> keypoints1, keypoints2;
    //Mat img_matches;
    //
    //vector< DMatch >good_matches;
    //for (size_t i = 0; i < keypoints_1.size(); i++) {
    //    keypoints1.push_back(KeyPoint(keypoints_1[i], 1.f));
    //}
    //for (size_t i = 0; i < keypoints_2.size(); i++) {
    //    keypoints2.push_back(KeyPoint(keypoints_2[i], 1.f));
    //}
    //for (size_t i = 0; i < match_kepoints.size(); i++) {
    //    int query_index = match_kepoints[i].first;
    //    int target_index = match_kepoints[i].second;
    //    //float distance = keypoints_1[query_index].x - keypoints_2[target_index].x
    //    DMatch tmp_match(query_index, target_index, 0);
    //    good_matches.push_back(tmp_match);
    //}

   
    //drawMatches(image_1, keypoints1, image_2, keypoints2, good_matches, img_matches, Scalar::all(-1),
    //    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //

    //vector<KeyPoint> match_list_1, match_list_2;
    //for (int i = 0;i < good_matches.size();i++) {
    //    KeyPoint source = keypoints1[good_matches[i].queryIdx];
    //    KeyPoint destination = keypoints2[good_matches[i].trainIdx];
    //    match_list_1.push_back(source);
    //    match_list_2.push_back(destination);
    //}
    //
    ////Mat h =  findHomography(match_list_1, match_list_2, 0);
    ////Mat Homorgraphic = transform(keypoints1,  keypoints2, good_matches);
    //Mat new_image;
    //new_image = transform_test(image_1, image_2, match_list_1, match_list_2);
   
    //auto total_end = chrono::steady_clock::now();
    //auto total_elapsed = chrono::duration<double>(total_end - total_begin);
    //cout << "total: " << total_elapsed.count() << " seconds" << endl;

    //namedWindow("my", 0);
    //resizeWindow("my", new_image.size() / 4);
    //imshow("my", new_image);
    //imwrite("result.jpg", new_image);
    //waitKey(0);
    //destroyAllWindows();
    return 0;
    
}