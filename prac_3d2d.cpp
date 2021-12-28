

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
void FindFeatureMatches(const cv::Mat& img1, const cv::Mat& img2,
                        std::vector<cv::KeyPoint>& keypoint1, std::vector<cv::KeyPoint>& keypoint2,
                        std::vector<cv::DMatch>& matches) {
    cv::Mat descriptor1, descriptor2;
    // declare
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // feature
    detector->detect(img1, keypoint1);
    detector->detect(img2, keypoint2);
    // descriptor
    descriptor->compute(img1, keypoint1, descriptor1);
    descriptor->compute(img2, keypoint2, descriptor2);
    std::vector<cv::DMatch> match;
    matcher->match(descriptor1, descriptor2, match);

    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < descriptor1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    for (int i = 0; i < descriptor1.rows; i++) {
        if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
void BundleAdjustment(const std::vector<cv::Point3f> points_3d,
                      const std::vector<cv::Point2f> points_2d,
                      const cv::Mat& K,
                      cv::Mat& R, cv::Mat& t) {
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    optimizer.addVertex(pose);

    int index = 1;
    for (const auto p : points_3d) {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }
    // typedef g2o::BlockSolver < g2o::BlockSover
    g2o::CameraParameters* camera = new g2o::CameraParameters(
        K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    index = 1;
    for (const cv::Point2f p : points_2d) {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    // cout << endl
    //  << "after optimization:" << endl;
    std::cout << "T=" << std::endl
              << Eigen::Isometry3d(pose->estimate()).matrix() << std::endl;
}

int main(int argc, char** argv) {
    // if (argc != 3) {
    //     printf("%d", (argc));
    //     printf("number");
    //     return -1;
    // }
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);

    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoint1, keypoint2;
    std::vector<cv::DMatch> matches;
    FindFeatureMatches(img1, img2, keypoint1, keypoint2, matches);

    cv::Mat depth1_img = cv::imread(argv[3], cv::IMREAD_UNCHANGED);  // or map

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (auto& m : matches) {
        auto d = depth1_img.ptr<unsigned short>(int(keypoint1[m.queryIdx].pt.y))[int(keypoint1[m.queryIdx].pt.x)];
        if (d == 0) continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoint1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoint2[m.trainIdx].pt);
    }
    cv::Mat r, t;
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);
    cv::Mat R;
    cv::Rodrigues(r, R);
    std::cout << "R:" << R << std::endl;
    std::cout << "t:" << t << std::endl;

    BundleAdjustment(pts_3d, pts_2d, K, R, t);

    // std::cout << 180.0 * std::atan2(R.at<float>(1, 2), R.at<float>(2, 2)) / M_PI;

    // X = atan(r21 / r11)
    //     Z = atan(r32 / r33)
    // [출처] 회전행렬에서 오일러 각 구하기|작성자 Luich
}
