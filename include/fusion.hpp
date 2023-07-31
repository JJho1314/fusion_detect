#ifndef _FUSION_H_
#define _FUSION_H_

#include <pcl/io/pcd_io.h>	   //PCL的PCD格式文件的输入输出头文件
#include <pcl/point_types.h>   //PCL对各种格式的点的支持头文件
#include <pcl/common/common.h> //getMinMax3D()函数所在头文件
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "common.hpp"
#include "render/box.h"

extern std::string save_dir;

class Fusion
{
private:
	float maxX = 250.0, maxY = 30.0, minZ = -4;
	cv::Mat Calib_P2;
	cv::Mat Calib_R_rect;
	cv::Mat Calib_Tr_velo_to_cam;
	float getIOU(const BBox2D &box1, const BBox2D &box2); // 计算IOU

public:
	Fusion()
	{
		// double_t cameraIn[12] = { 9.9211942488036937e+02, 0., 9.5350126304636274e+02, 0.,
		//                           0., 9.9011738879895415e+02, 5.4450876344343885e+02, 0.,
		// 						  0., 0., 1., 0.};

		// double_t RT[16] = { 0.01563570,-0.02261498, 0.99962194,0.06089430,
		// 				   -0.99971406,-0.01841716, 0.01522058,0.12742106,
		// 					0.01806581,-0.99957419,-0.02289650,0.10774996,
		// 					0.00000000, 0.00000000, 0.00000000,1.00000000  };

		// double_t RT[16] = { 0.01563580166140476151,-0.99971500487122299094	, 0.018066000540947798709, 0.12448600358583921328,
		// 				   -0.022615001453936586,-0.01841699960916738031, -0.99957500239766823121, 0.1114280048206048,
		// 					0.9996220025846791906,0.015220498085349800543,-0.0228964995029834145, -0.060343597206168989297,
		// 					0.00000000, 0.00000000, 0.00000000,1.00000000  };

		// cv::Mat(4, 4, 6, &cameraIn).copyTo(CameraMat);

		// cv::Mat(4, 4, 6, &RT).copyTo(CameraExtrinsicMat);

		cv::eigen2cv(P2, Calib_P2);
		cv::eigen2cv(R_rect, Calib_R_rect);
		cv::eigen2cv(Tr_velo_to_cam, Calib_Tr_velo_to_cam);
	}

	// 将全部点云投影到图像上进行可视化
	cv::Mat pointcloud2_to_image(const cv::Mat &raw_image, const pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud);
	// 投影聚类后的点云图像上可视化
	cv::Mat cluster_to_image(const cv::Mat &raw_image, const pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud, const std::vector<std::pair<int, int>> matchedPairs, const std::vector<pcl::PointCloud<pcl::PointXYZ>> cloudClusters);

	void draw_projected_box3d(const cv::Mat &raw_image, const std::vector<pcl::PointCloud<pcl::PointXYZ>> boxes, const std::vector<std::pair<int, int>> matchedPairs);
	// 计算雷达3D包围框投影到图像的最大包围面积
	void convert_3DBox_to_2DBox(const cv::Mat &raw_image, const std::vector<pcl::PointCloud<pcl::PointXYZ>> &boxes3d, std::vector<BBox2D> &boxes2d);
	// 匈牙利匹配算法
	std::vector<std::pair<int, int>> hungarianAlgorithm(const std::vector<std::vector<float>> &costMatrix, std::vector<float> threshold);
	// 雷达2D包围框和相机2D包围框进行数据关联
	std::tuple<std::vector<std::pair<int, int>>, std::vector<BBox2D>, std::vector<BBox2D>> associate(const std::vector<BoxQ> &BBoxs, const std::vector<BBox2D> &lidarBoxes, const std::vector<BBox2D> &cameraBoxes);
	// 建立3D包围框的融合关系
	void build_fused_object(const cv::Mat &raw_image, const pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud, const std::vector<BBox2D> &lidarBoxes, const std::vector<std::pair<int, int>> matchedPairs, const std::vector<pcl::PointCloud<pcl::PointXYZ>> cloudClusters);

	void lidar_to_cam(const Eigen::Vector3f &lidar_point, Eigen::Vector3f &cam_point);

	// void lidar_to_cam(const Eigen::Matrix3f &lidar_rotation_matrix, Eigen::Matrix3f &camera_rotation_matrix);
};

#endif