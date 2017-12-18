#include <fstream>
#include <filesystem>
#include "foundation/logging.h"
#include "foundation/testing.h"
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace vision
{
    const cv::Vec3b black(0, 0, 0);
    const cv::Vec3b white(255, 255, 255);
    const cv::Vec3b red(0, 0, 255);
    const cv::Vec3b lime(0, 255, 0);
    const cv::Vec3b blue(255, 0, 0);
    const cv::Vec3b yellow(0, 255, 255);
    const cv::Vec3b cyan(255, 255, 0);
    const cv::Vec3b magenta(255, 0, 255);
    const cv::Vec3b silver(192, 192, 192);
    const cv::Vec3b grey(128, 128, 128);
    const cv::Vec3b maroon(0, 0, 128);
    const cv::Vec3b green(0, 128, 0);
    const cv::Vec3b navy(128, 0, 0);
    const cv::Vec3b olive(0, 128, 128);
    const cv::Vec3b teal(128, 128, 0);
    const cv::Vec3b purple(128, 0, 128);

    static int GetMaxColorIndex()
    {
        return 16;
    }

    static cv::Vec3b GetColor(int index)
    {
        auto warp_around_index = index % GetMaxColorIndex();
        switch (warp_around_index)
        {
        case 0:            return black;
        case 1:            return white;
        case 2:            return red;
        case 3:            return lime;
        case 4:            return blue;
        case 5:            return yellow;
        case 6:            return cyan;
        case 7:            return magenta;
        case 8:            return silver;
        case 9:            return grey;
        case 10:            return maroon;
        case 11:            return olive;
        case 12:            return green;
        case 13:            return purple;
        case 14:            return teal;
        case 15:            return navy;
        default:            return black;
        }
    }

    cv::Mat ColorLabelImage(const cv::Mat & label_image)
    {
        if (label_image.type() != CV_32S)
        {
            throw std::invalid_argument("Wrong matrix type");
        }
        cv::Mat colored_image(label_image.size(), CV_8UC3);

        for (int i = 0; i < label_image.rows; ++i)
        {
            for (int j = 0; j < label_image.cols; ++j)
            {
                int32_t label = label_image.at< int32_t >(i, j);
                auto label_color = GetColor(label);
                colored_image.at< cv::Vec3b >(i, j) = label_color;
            }
        }
        return colored_image;
    }

    cv::Mat LoadBinaryMatFile(const std::string & image_filename)
    {
        std::ifstream fs(image_filename, std::fstream::binary);

        // Header
        int rows, cols, type, channels;
        fs.read((char *)&rows, sizeof(int)); // rows
        fs.read((char *)&cols, sizeof(int)); // cols
        fs.read((char *)&type, sizeof(int)); // type
        fs.read((char *)&channels, sizeof(int)); // channels

        // Data
        cv::Mat mat(rows, cols, type);
        fs.read((char *)mat.data, CV_ELEM_SIZE(type) * rows * cols);
        return mat;
    }

    inline float Distance(cv::Vec3f point_a, cv::Vec3f point_b)
    {
        return abs(point_b.val[2] - point_a.val[2]);
        //return cv::norm(point_a, point_b);
        /*float distance = 0;
        for (int i = 0; i < 3; i++)
        {
            const float dist = point_a.val[i] - point_b.val[i];
            distance += dist * dist;
        }
        return sqrt(distance);*/
    }

    bool MatchLabeledImages(const cv::Mat & label_image_1, const cv::Mat & label_image_2)
    {
        std::map<int, int> label_map;

        for (int x = 0; x < label_image_1.rows; x++)
        {
            for (int y = 0; y < label_image_1.cols; y++)
            {
                auto label_1 = label_image_1.at<int>(x, y);
                auto label_2 = label_image_2.at<int>(x, y);
                auto label_match = label_map.find(label_1);
                if (label_match != label_map.end())
                {
                    if(label_2 != label_match->second)
                    {
                        return false;
                    }
                }
                else
                {
                    label_map[label_1] = label_2;
                }
            }
        }
        return true;
    }

    struct {
        bool operator()(std::pair<float, cv::Point> a, std::pair<float, cv::Point> b) const
        {
            return a.first < b.first;
        }
    } ComparePixelDistance;

    void FindClosestPoint(const cv::Mat& point_cloud, const cv::Mat & label_image, int x, int y, float& min_distance, cv::Point& nearest_pixel, int filter_size)
    {
        std::vector<std::pair<float, cv::Point>> distances;
        int start_filter_limit = -filter_size / 2;
        int end_filter_limit = filter_size / 2;

        auto point = point_cloud.at<cv::Vec3f>(x, y);
        auto label = label_image.at<int>(x, y);
        min_distance = std::numeric_limits<float>::infinity();
        nearest_pixel = cv::Point(x, y);
        for (int i = start_filter_limit; i <= end_filter_limit; i++)
        {
            for (int j = start_filter_limit; j <= end_filter_limit; j++)
            {
                auto x_index = x + i;
                auto y_index = y + j;
                //std::cout << x_index << ", " << y_index << std::endl;
                if (x_index == x && y_index == y)
                {
                    //std::cout << "skipping same pixel" << std::endl;
                    continue;
                }
                if (label_image.at<int>(x_index, y_index) <= 0)
                {
                    //std::cout << "skipping neighbour" << std::endl;
                    continue;
                }
                auto neighbour_point = point_cloud.at<cv::Vec3f>(x_index, y_index);
                float distance = Distance(neighbour_point, point);
                // keep track of distances within cutoff
                if (distance < 0.2)
                {
                    distances.push_back(std::make_pair(distance, cv::Point(x_index, y_index)));
                }
                        
                if (min_distance > distance)
                {
                    min_distance = distance;
                    nearest_pixel.x = x_index;
                    nearest_pixel.y = y_index;
                }
            }
        }
        //std::cout << label << std::endl;
        //std::cout << distances.size() << "-> " << distances[0].first  << ", "<< distances[0].second << ", " << distances[1].first << ", " << distances[1].second <<std::endl;
        std::sort(distances.begin(), distances.end(), ComparePixelDistance);
        //std::cout << distances.size() << "-> " << distances[0].first << ", " << distances[0].second << ", " << distances[1].first << ", " << distances[1].second << std::endl;
        min_distance = distances[0].first;
        nearest_pixel.x = distances[0].second.x;
        nearest_pixel.y = distances[0].second.y;
         }

    cv::Mat ConnectedPointCloud(const cv::Mat& point_cloud, int filter_size = 3)
    {
        int step_size = 1;
        float distance = 0.0;
        int labels = 1;
        int start_filter_limit = -filter_size / 2;
        int end_filter_limit = filter_size / 2;

        cv::Mat label_image = cv::Mat(point_cloud.size(), CV_32SC1, cv::Scalar(-1));
        cv::Mat distances = cv::Mat::zeros(point_cloud.size(), CV_32FC1);
        cv::Vec3f zero_point = cv::Vec3f(0, 0, 0);

        std::vector<cv::Mat> xyz_channels;
        cv::split(point_cloud, xyz_channels);
        cv::medianBlur(xyz_channels[2], xyz_channels[2], filter_size);

        for (int x = 0; x < point_cloud.rows; x++)
        {
            for (int y = 0; y < point_cloud.cols; y++)
            {
                auto point = point_cloud.at<cv::Vec3f>(x, y);
                //std::cout << x << ", " << y << " = " << (point.val[2] < 0.01) << ", " << (x < -start_filter_limit) << ", " << (y < -start_filter_limit) << ", " << (x > point_cloud.rows - end_filter_limit) << ", " << (y > point_cloud.cols - end_filter_limit) << std::endl;
                if (point.val[2] < 0.01 || x < -start_filter_limit || y < -start_filter_limit || x >= point_cloud.rows - end_filter_limit || y >= point_cloud.cols - end_filter_limit)
                {
                    label_image.at<int>(x, y) = 0;
                    continue;
                }
            }
        }
        for (int x = point_cloud.rows; x > 0; x--)
        {
            for (int y = point_cloud.rows; y > 0; y--)
            {
                if (label_image.at<int>(x, y) == 0)
                    continue;
                float min_distance;
                cv::Point nearest_pixel;

                FindClosestPoint(point_cloud, label_image, x, y, min_distance, nearest_pixel, filter_size);

                int current_label = -1;
                if (min_distance < 0.02)
                {
                    //std::cout << "close pixel at " << min_distance << " == " << nearest_pixel << " , " << cv::Point(x, y) << std::endl;
                    current_label = label_image.at<int>(nearest_pixel.x, nearest_pixel.y);
                }
                if (current_label == -1)
                {
                    current_label = ++labels;
                    std::cout << "label == " << min_distance << " == " << labels << " at" << nearest_pixel << " , " << cv::Point(x, y) << std::endl;
                }
                label_image.at<int>(x, y) = current_label;
                //label_image.at<int>(nearest_pixel.x, nearest_pixel.y) = current_label;
                if (min_distance < 100)
                {
                    distances.at<float>(x, y) = min_distance;
                }

            }
        }
        // second pass

        cv::imshow("distances", distances);
        return label_image.clone();
    }
}

/// Meta_programing
/// Modular
/// create opencv implementation
/// add benchmark porject
/// create halide implementation 
/// document
namespace testing
{
    cv::Mat CreateXData()
    {
        float x_data[49] = { -3, -2, -1, 0, 1, 2, 3,
            -3, -2, -1, 0, 1, 2, 3,
            -3, -2, -1, 0, 1, 2, 3,
            -3, -2, -1, 0, 1, 2, 3,
            -3, -2, -1, 0, 1, 2, 3,
            -3, -2, -1, 0, 1, 2, 3,
            -3, -2, -1, 0, 1, 2, 3 };
        return cv::Mat(7, 7, CV_32FC1, x_data).clone();
    }

    cv::Mat CreateYData()
    {
        float y_data[49] = { -3, -3, -3, -3, -3, -3, -3,
            -2, -2, -2, -2, -2, -2, -2,
            -1, -1, -1, -1, -1, -1, -1,
            0,  0,  0,  0,  0,  0,  0,
            1,  1,  1,  1,  1,  1,  1,
            2,  2,  2,  2,  2,  2,  2,
            3,  3,  3,  3,  3,  3,  3 };
        return cv::Mat(7, 7, CV_32FC1, y_data).clone();
    }

    cv::Mat CreateZData(int test_case )
    {
        cv::Mat z_data;
        float z_data_1[49] = { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.81,  0.82,  0.51,  0.81,  0.81,  0.0,
            0.0,  0.82,  0.81,  0.53,  0.52,  0.82,  0.0,
            0.0,  0.83,  0.83,  0.52,  0.53,  0.82,  0.0,
            0.0,  0.84,  0.84,  0.51,  0.54,  0.81,  0.0,
            0.0,  0.85,  0.52,  0.53,  0.52,  0.82,  0.0,
            0.0,  0.86,  0.00,  0.00,  0.00,  0.83,  0.0 };
        float z_data_2[49] = { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                               0.0,  0.81,  0.505,  0.525,  0.54,  0.81,  0.0,
                               0.0,  0.82,  0.55,  0.52,  0.56,  0.81,  0.0,
                               0.0,  0.835,  0.825,  0.82,  0.84,  0.83,  0.0,
                               0.0,  0.84,  0.52,  0.52,  0.53,  0.82,  0.0,
                               0.0,  0.85,  0.53,  0.52,  0.54,  0.82,  0.0,
                               0.0,  0.00,  0.00,  0.00,  0.00,  0.80,  0.0 };
        switch(test_case)
        {
        case 1:
            z_data = cv::Mat(7, 7, CV_32FC1, z_data_1);
            break;
        case 2:
            z_data = cv::Mat(7, 7, CV_32FC1, z_data_2);
            break;
        default:
            throw std::exception("test case not defined for z data");
        }
        return z_data.clone();
    }

    cv::Mat CreateExpectedLabelData(int test_case)
    {
        cv::Mat labels_image;
        int32_t labels_data_1[49] = { 0,  0,  0,  0,  0,  0,  0,
            0,  1,  1,  2,  3,  3,  0,
            0,  1,  1,  2,  2,  3,  0,
            0,  1,  1,  2,  2,  3,  0,
            0,  1,  1,  2,  2,  3,  0,
            0,  1,  2,  2,  2,  3,  0,
            0,  0,  0,  0,  0,  0,  0 };
        int32_t labels_data_2[49] = { 0,  0,  0,  0,  0,  0,  0,
            0,  1,  2,  2,  2,  1,  0,
            0,  1,  2,  2,  2,  1,  0,
            0,  1,  1,  1,  1,  1,  0,
            0,  1,  3,  3,  3,  1,  0,
            0,  1,  3,  3,  3,  1,  0,
            0,  0,  0,  0,  0,  0,  0 };
        switch (test_case)
        {
        case 1:
            labels_image = cv::Mat(7, 7, CV_32SC1, labels_data_1);
            break;
        case 2:
            labels_image = cv::Mat(7, 7, CV_32SC1, labels_data_2);
            break;
        default:
            throw std::exception("test case not defined for expected labeled data");
        }
        return labels_image.clone();
        
    }

    cv::Mat CreatePointCloudTestData(int test_case)
    {
        cv::Mat x_channel = CreateXData();
        cv::Mat y_channel = CreateYData();
        cv::Mat z_channel = CreateZData(test_case);

        std::vector<cv::Mat> xyz_channels = { x_channel, y_channel, z_channel };
        cv::Mat pointcloud;
        cv::merge(xyz_channels, pointcloud);
        return pointcloud;
    }

    //TEST(MatchLabel, IdentiticalImages)
    //{   
    //    cv::Mat label_image_1 = CreateExpectedLabelData(1);
    //    cv::Mat label_image_2 = CreateExpectedLabelData(1);
    //    ASSERT_TRUE(vision::MatchLabeledImages(label_image_1, label_image_2));
    //}

    //TEST(MatchLabel, SimilarImages)
    //{
    //    cv::Mat label_image_1 = CreateExpectedLabelData(1);
    //    cv::Mat label_image_2 = 4 * CreateExpectedLabelData(1);
    //    ASSERT_TRUE(vision::MatchLabeledImages(label_image_1, label_image_2));
    //}

    //TEST(MatchLabel, DifferentLabelsImages)
    //{
    //    cv::Mat label_image_1 = CreateExpectedLabelData(1);
    //    cv::Mat label_image_2 = CreateExpectedLabelData(2);
    //    ASSERT_FALSE(vision::MatchLabeledImages(label_image_1, label_image_2));
    //}

    TEST(FindClosesetPoint, DifferentLabelsImages)
    {
        cv::Mat pointcloud = CreatePointCloudTestData(2);
        int32_t labels_data[49] = { 0,  0,  0,  0,  0,  0,  0,
                                    0,  1,  2,  2,  2,  1,  0,
                                    0,  1,  2,  2,  0,  0,  0,
                                    0,  1,  3,  3,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0 };
        cv::Mat labels_image = cv::Mat(7, 7, CV_32SC1, labels_data);
        float min_distance;
        cv::Point nearest_point;
        vision::FindClosestPoint(pointcloud, labels_image, 2, 3, min_distance, nearest_point, 3);
        float expected_min_distance = 0.005f;
        ASSERT_FLOAT_EQ(expected_min_distance, min_distance);
    }


    TEST(SampleTest, DISABLED_TestCase1)
    {   
        cv::Mat pointcloud = CreatePointCloudTestData(1);
        cv::Mat expected_label = CreateExpectedLabelData(1);
        cv::Mat labels = vision::ConnectedPointCloud(pointcloud);

        //bool match = MatchLabeledImages(expected_label, labels);
        ASSERT_TRUE(vision::MatchLabeledImages(expected_label, labels));

        //auto expected_color_label_image = vision::ColorLabelImage(expected_label);
        //cv::imshow("expected label image", expected_color_label_image);

        //auto color_label_image = vision::ColorLabelImage(labels);
        //cv::imshow("label_image", color_label_image);
        //cv::waitKey(-1);
        
    }

    TEST(SampleTest, DISABLED_TestCase2)
    {
        cv::Mat pointcloud = CreatePointCloudTestData(2);
        cv::Mat expected_label = CreateExpectedLabelData(2);
        cv::Mat labels = vision::ConnectedPointCloud(pointcloud);

        ASSERT_TRUE(vision::MatchLabeledImages(expected_label, labels));

        auto expected_color_label_image = vision::ColorLabelImage(expected_label);
        cv::imshow("expected label image", expected_color_label_image);

        auto color_label_image = vision::ColorLabelImage(labels);
        cv::imshow("label_image", color_label_image);
        cv::waitKey(-1);
    }


    //TEST(SampleTest, TestCase3)
    //{
    //    namespace fs = std::experimental::filesystem;

    //    std::string path = "C:\\Dev\\data\\test_data";
    //    for (auto & p : fs::directory_iterator(path))   
    //    {
    //        cv::Mat image = vision::LoadBinaryMatFile(p.path().string());
    //        cv::Mat labels_image = vision::ConnectedPointCloud(image);
    //        //cv::imshow("test_image", labels_image);
    //        //cv::waitKey(-1);
    //    }

    //    ASSERT_TRUE(true);
    //}

}
