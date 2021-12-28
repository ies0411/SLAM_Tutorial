#include <iostream>
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

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("%d", (argc));
        printf("number");
        return -1;
    }
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    // cv::imshow("display", img1);
    // cv::waitKey(0);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoint1, keypoint2;

    // cv::Mat outimg1;
    // cv::drawKeypoints(img1, keypoint1, outimg1, cv::Scalar::all(-1), 0);
    // printf("check");
    // cv::imshow("1", outimg1);
    // cv::waitKey(0);
    // matching

    // std::vector<cv::DMatch> matches;
    // cv::Mat img_matched;
    // matcher->match(descriptor1, descriptor2, matches);
    // cv::drawMatches(img1, keypoint1, img2, keypoint2, matches, img_matched);
    // imshow("matched", img_matched);
    // cv::waitKey(0);
    // std::cout << descriptor1 << std::endl;
    // double_t min_dist = 10000, max_dist = 0;
    // for (int i = 0; i < descriptor1.rows; i++) {
    //     double_t dist = matches[i].distance;
    //     min_dist = std::min(dist, min_dist);
    //     max_dist = std::max(dist, max_dist);
    // }
    // std::cout << min_dist << " , " << max_dist << std::endl;
    // min_dist = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    // max_dist = std::max_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;

    // printf("-- Max dist : %f \n", max_dist);
    // printf("-- Min dist : %f \n", min_dist);
}
