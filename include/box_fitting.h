
#ifndef MY_PCL_TUTORIAL_BOX_FITTING_H
#define MY_PCL_TUTORIAL_BOX_FITTING_H

#include <array>
#include <pcl/io/pcd_io.h>
#include <vector>
#include "render/box.h"

// #include "component_clustering.h"

using namespace std;
using namespace pcl;

class Box_fitting
{
    public:
        //  将最小面积矩形（MAR）[128]应用于每个聚类对象，从而生成一个2D框，当与保留在聚类过程中的高度信息结合后，它便成为3D边界框
        void getBoundingBox(std::vector<PointCloud<PointXYZ>> &clusteredPoints, std::vector<PointCloud<PointXYZ>> &bbPoints, bool filter);

        // Function to calculate the bounding box center, dimensions, and orientation quaternion
        void calculateBoundingBoxes(const std::vector<pcl::PointCloud<pcl::PointXYZ>> &boundingBoxes, std::vector<BoxQ> &BBoxes);

    private:
        // float picScale = 30;
        int roiM = 30;
        float picScale = 900 / roiM; // ???
        int ramPoints = 80;
        int lSlopeDist = 1.0;
        ////////////////////////int lSlopeDist = 3.0;
        ////////////////////////int lnumPoints = 300;
        int lnumPoints = 5;
        //////////////////////////////float sensorHeight = 1.73;
        float sensorHeight = 1.6; // 激光雷达距离地面高度
        // float tHeightMin = 1.2;
        float tHeightMin = 0.0;
        float tHeightMax = 2.0;
        // float tWidthMin = 0.5;
        // float tWidthMin = 0.4;
        float tWidthMin = 0.0; // 0.25
        float tWidthMax = 3.5;
        float tLenMin = 0.0; // 0.5
        float tLenMax = 14.0;
        float tAreaMax = 10.0;
        // float tRatioMin = 1.3;
        // float tRatioMax = 5.0;

        float tRatioMin = 1;
        float tRatioMax = 8.0;

        float minLenRatio = 3.0;
        float tPtPerM3 = 8;

        void getPointsInPcFrame(cv::Point2f rectPoints[], vector<cv::Point2f> &pcPoints, int offsetX, int offsetY);

        bool ruleBasedFilter(std::vector<cv::Point2f> pcPoints, float maxZ, int numPoints); // 规则滤除
};

#endif // MY_PCL_TUTORIAL_BOX_FITTING_H