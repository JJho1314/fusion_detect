#include <iostream>			   //标准输入输出流
#include <pcl/io/pcd_io.h>	   //PCL的PCD格式文件的输入输出头文件
#include <pcl/point_types.h>   //PCL对各种格式的点的支持头文件
#include <pcl/common/common.h> //getMinMax3D()函数所在头文件
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm> //max,min

#include "fusion.hpp"
#include "render/box.h"
#include <Eigen/Dense>

cv::Mat Fusion::pointcloud2_to_image(const cv::Mat &raw_image, const pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud)
{
	int w = raw_image.cols;
	int h = raw_image.rows;
	double *distance = new double[w * h * sizeof(double)];

	cv::Mat visImg = raw_image.clone();
	cv::Mat overlay = visImg.clone();

	cv::Mat color_map_;
	// set color map
	cv::Mat gray_scale(256, 1, CV_8UC1);

	for (int i = 0; i < 256; i++)
	{
		gray_scale.at<uchar>(i) = i;
	}

	cv::applyColorMap(gray_scale, color_map_, cv::COLORMAP_JET);

	pcl::PointXYZI min;						 // xyz的最小值
	pcl::PointXYZI max;						 // xyz的最大值
	pcl::getMinMax3D(*pointcloud, min, max); // 获取所有点中的坐标最值

	float distance_range = max.x - min.x;

	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::copyPointCloud(*pointcloud, *cloud);

	pcl::PointCloud<pcl::PointXYZI>::const_iterator it;

	for (it = cloud->points.begin(); it != cloud->points.end(); it++)
	{
		if (it->x < 0.0)
		{
			continue;
		}

		X.at<double>(0, 0) = it->x;
		X.at<double>(1, 0) = it->y;
		X.at<double>(2, 0) = it->z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		cv::Point pt;
		pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		// std::cout << "x: " << pt.x << ", " << "y: " << pt.y ;

		float val = it->x;
		int color_id = distance_range ? ((val - min.x) * 255 / distance_range) : 128;

		// Divide color into each element
		cv::Vec3b color = color_map_.at<cv::Vec3b>(color_id);
		int red = color[0];
		int green = color[1];
		int blue = color[2];

		// float maxVal = 20.0;
		// int red = std::min(255, (int)(255 * abs((val - maxVal) / maxVal)));
		// int green = std::min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal))));
		cv::circle(overlay, pt, 1, cv::Scalar(red, green, blue), CV_FILLED);
	}

	cv::imwrite(save_dir + "fusion.jpg", overlay);
	// delete[] distance;
	return overlay;
}

cv::Mat Fusion::cluster_to_image(const cv::Mat &raw_image, const pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud, const std::vector<std::pair<int, int>> matchedPairs, const std::vector<pcl::PointCloud<pcl::PointXYZ>> cloudClusters)
{
	int w = raw_image.cols;
	int h = raw_image.rows;

	cv::Mat overlay = raw_image.clone();

	cv::Mat color_map_;
	// set color map
	cv::Mat gray_scale(256, 1, CV_8UC1);

	for (int i = 0; i < 256; i++)
	{
		gray_scale.at<uchar>(i) = i;
	}

	cv::applyColorMap(gray_scale, color_map_, cv::COLORMAP_JET);

	pcl::PointXYZI min;						 // xyz的最小值
	pcl::PointXYZI max;						 // xyz的最大值
	pcl::getMinMax3D(*pointcloud, min, max); // 获取所有点中的坐标最值

	float distance_range = max.x - min.x;

	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);

	for (const auto &match : matchedPairs)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(cloudClusters[match.first], *cloud);

		pcl::PointCloud<pcl::PointXYZ>::const_iterator it;

		for (it = cloud->points.begin(); it != cloud->points.end(); it++)
		{
			if (it->x < 0.0)
			{
				continue;
			}

			X.at<double>(0, 0) = it->x;
			X.at<double>(1, 0) = it->y;
			X.at<double>(2, 0) = it->z;
			X.at<double>(3, 0) = 1;

			Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

			cv::Point pt;
			pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
			pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

			// std::cout << "x: " << pt.x << ", " << "y: " << pt.y ;

			float val = it->x;
			int color_id = distance_range ? ((val - min.x) * 255 / distance_range) : 128;

			// Divide color into each element
			cv::Vec3b color = color_map_.at<cv::Vec3b>(color_id);
			int red = color[0];
			int green = color[1];
			int blue = color[2];

			cv::circle(overlay, pt, 1, cv::Scalar(red, green, blue), CV_FILLED);
		}
	}

	cv::imwrite(save_dir + "cluster_to_img.jpg", overlay);
	// delete[] distance;
	return overlay;
}

void Fusion::convert_3DBox_to_2DBox(const cv::Mat &raw_image, const std::vector<pcl::PointCloud<pcl::PointXYZ>> &boxes3d, std::vector<BBox2D> &boxes2d)
{
	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);

	cv::Mat visImg = raw_image.clone();
	cv::Point pt;

	for (int i = 0; i < boxes3d.size(); i++)
	{
		std::vector<int> pts_2d_x;
		std::vector<int> pts_2d_y;
		for (int j = 0; j < boxes3d[i].size(); j++)
		{
			X.at<double>(0, 0) = boxes3d[i][j].x;
			X.at<double>(1, 0) = boxes3d[i][j].y;
			X.at<double>(2, 0) = boxes3d[i][j].z;
			X.at<double>(3, 0) = 1;

			Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

			pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
			pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

			pts_2d_x.emplace_back(pt.x);
			pts_2d_y.emplace_back(pt.y);
		}
		auto x_max = max_element(pts_2d_x.begin(), pts_2d_x.end());
		auto y_max = max_element(pts_2d_y.begin(), pts_2d_y.end());
		auto x_min = min_element(pts_2d_x.begin(), pts_2d_x.end());
		auto y_min = min_element(pts_2d_y.begin(), pts_2d_y.end());

		float x1 = std::max(*x_min, 0);
		float y1 = std::max(*y_min, 0);
		float x2 = std::min(*x_max, raw_image.cols);
		float y2 = std::min(*y_max, raw_image.rows);

		BBox2D box = {x1, y1, x2, y2};

		cv::line(visImg, cv::Point(x1, y1), cv::Point(x2, y1), cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(visImg, cv::Point(x2, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(visImg, cv::Point(x2, y2), cv::Point(x1, y2), cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(visImg, cv::Point(x1, y2), cv::Point(x1, y1), cv::Scalar(255, 0, 0), 2, CV_AA);

		boxes2d.emplace_back(box);
	}

	cv::imwrite(save_dir + "3D_to_2D_img.jpg", visImg);
}

// Function to calculate the Intersection over Union (IOU) between two 2D bounding boxes
float Fusion::getIOU(const BBox2D &box1, const BBox2D &box2)
{
	int xA = std::max(box1.x_min, box2.x_min);
	int yA = std::max(box1.y_min, box2.y_min);
	int xB = std::min(box1.x_max, box2.x_max);
	int yB = std::min(box1.y_max, box2.y_max);

	int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);

	int box1Area = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min);
	int box2Area = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min);

	float iou = static_cast<float>(interArea) / (box1Area + box2Area - interArea);
	return iou;
}

// Function to find the best matches using the Hungarian Algorithm
std::vector<std::pair<int, int>> Fusion::hungarianAlgorithm(const std::vector<std::vector<float>> &costMatrix, std::vector<float> threshold)
{
	int numLidarBoxes = costMatrix.size();
	int numCameraBoxes = costMatrix[0].size();

	std::vector<int> hungarianRow(numLidarBoxes, -1);
	std::vector<int> hungarianCol(numCameraBoxes, -1);

	for (int i = 0; i < numLidarBoxes; ++i)
	{
		for (int j = 0; j < numCameraBoxes; ++j)
		{
			if (costMatrix[i][j] > threshold[i])
			{
				if (hungarianRow[i] == -1 && hungarianCol[j] == -1)
				{
					hungarianRow[i] = j;
					hungarianCol[j] = i;
				}
			}
		}
	}

	std::vector<std::pair<int, int>> matchedPairs;
	for (int i = 0; i < numLidarBoxes; ++i)
	{
		int j = hungarianRow[i];
		if (j != -1)
		{
			matchedPairs.emplace_back(i, j);
		}
	}

	return matchedPairs;
}

// Call the Hungarian Algorithm to find the best matching pairs
// Function to associate LiDAR and camera bounding boxes using the Hungarian Algorithm
std::tuple<std::vector<std::pair<int, int>>, std::vector<BBox2D>, std::vector<BBox2D>> Fusion::associate(
	const std::vector<BoxQ> &BBoxs, const std::vector<BBox2D> &lidarBoxes, const std::vector<BBox2D> &cameraBoxes)
{

	int numLidarBoxes = lidarBoxes.size();
	int numCameraBoxes = cameraBoxes.size();
	std::vector<float> thresholds;

	// Define a new IOU Matrix nxm with old and new boxes
	std::vector<std::vector<float>> iouMatrix(numLidarBoxes, std::vector<float>(numCameraBoxes, 0.0f));

	// Calculate the IOU value for each pair of boxes
	for (int i = 0; i < numLidarBoxes; ++i)
	{
		float threshold = 0.5;
		for (int j = 0; j < numCameraBoxes; ++j)
		{
			iouMatrix[i][j] = getIOU(lidarBoxes[i], cameraBoxes[j]);
			// std::cout << "IOU: " << getIOU(lidarBoxes[i], cameraBoxes[j]) << std::endl;
			// printf("(%d %d %d %d)\n", int(lidarBoxes[i].x_min), int(lidarBoxes[i].y_min), int(lidarBoxes[i].x_max), int(lidarBoxes[i].y_max));
			// printf("(%d %d %d %d)\n", int(cameraBoxes[j].x_min), int(cameraBoxes[j].y_min), int(cameraBoxes[j].x_max), int(cameraBoxes[j].y_max));
		}

		if (BBoxs[i].bboxTransform(0) < 20)
		{
			threshold = 0.5;
		}
		else if (BBoxs[i].bboxTransform(0) >= 20 && BBoxs[i].bboxTransform(0) < 30)
		{
			threshold = 0.2;
		}
		else if (BBoxs[i].bboxTransform(0) >= 30 && BBoxs[i].bboxTransform(0) < 50)
		{
			threshold = 0.15;
		}
		else if (BBoxs[i].bboxTransform(0) >= 50)
		{
			threshold = 0.1;
		}

		thresholds.push_back(threshold);
	}

	// Call the Hungarian Algorithm to find the best matching pairs
	std::vector<std::pair<int, int>> matchedPairs = hungarianAlgorithm(iouMatrix, thresholds);

	// Create new unmatched lists for old and new boxes
	std::vector<BBox2D> unmatchedLidarBoxes;
	std::vector<BBox2D> unmatchedCameraBoxes;

	// Find unmatched LiDAR and camera boxes
	std::vector<bool> isLidarMatched(numLidarBoxes, false);
	std::vector<bool> isCameraMatched(numCameraBoxes, false);

	for (const auto &match : matchedPairs)
	{
		isLidarMatched[match.first] = true;
		isCameraMatched[match.second] = true;
	}

	for (int i = 0; i < numLidarBoxes; ++i)
	{
		if (!isLidarMatched[i])
		{
			unmatchedLidarBoxes.push_back(lidarBoxes[i]);
		}
	}

	for (int j = 0; j < numCameraBoxes; ++j)
	{
		if (!isCameraMatched[j])
		{
			unmatchedCameraBoxes.push_back(cameraBoxes[j]);
		}
	}

	return std::make_tuple(matchedPairs, unmatchedLidarBoxes, unmatchedCameraBoxes);
}

void Fusion::build_fused_object(const cv::Mat &raw_image, const pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud, const std::vector<BBox2D> &lidarBoxes, const std::vector<std::pair<int, int>> matchedPairs, const std::vector<pcl::PointCloud<pcl::PointXYZ>> cloudClusters)
{
	int w = raw_image.cols;
	int h = raw_image.rows;

	cv::Mat overlay = raw_image.clone();

	cv::Mat color_map_;
	// set color map
	cv::Mat gray_scale(256, 1, CV_8UC1);

	for (int i = 0; i < 256; i++)
	{
		gray_scale.at<uchar>(i) = i;
	}

	cv::applyColorMap(gray_scale, color_map_, cv::COLORMAP_JET);

	pcl::PointXYZI min;						 // xyz的最小值
	pcl::PointXYZI max;						 // xyz的最大值
	pcl::getMinMax3D(*pointcloud, min, max); // 获取所有点中的坐标最值

	float distance_range = max.x - min.x;

	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);

	for (const auto &match : matchedPairs)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(cloudClusters[match.first], *cloud);

		pcl::PointCloud<pcl::PointXYZ>::const_iterator it;

		for (it = cloud->points.begin(); it != cloud->points.end(); it++)
		{
			if (it->x < 0.0)
			{
				continue;
			}

			X.at<double>(0, 0) = it->x;
			X.at<double>(1, 0) = it->y;
			X.at<double>(2, 0) = it->z;
			X.at<double>(3, 0) = 1;

			Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

			cv::Point pt;
			pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
			pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

			// std::cout << "x: " << pt.x << ", " << "y: " << pt.y ;

			float val = it->x;
			int color_id = distance_range ? ((val - min.x) * 255 / distance_range) : 128;

			// Divide color into each element
			cv::Vec3b color = color_map_.at<cv::Vec3b>(color_id);
			int red = color[0];
			int green = color[1];
			int blue = color[2];

			cv::circle(overlay, pt, 1, cv::Scalar(red, green, blue), CV_FILLED);
		}

		cv::line(overlay, cv::Point(lidarBoxes[match.first].x_min, lidarBoxes[match.first].y_min), cv::Point(lidarBoxes[match.first].x_max, lidarBoxes[match.first].y_min), cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(overlay, cv::Point(lidarBoxes[match.first].x_max, lidarBoxes[match.first].y_min), cv::Point(lidarBoxes[match.first].x_max, lidarBoxes[match.first].y_max), cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(overlay, cv::Point(lidarBoxes[match.first].x_max, lidarBoxes[match.first].y_max), cv::Point(lidarBoxes[match.first].x_min, lidarBoxes[match.first].y_max), cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(overlay, cv::Point(lidarBoxes[match.first].x_min, lidarBoxes[match.first].y_max), cv::Point(lidarBoxes[match.first].x_min, lidarBoxes[match.first].y_min), cv::Scalar(255, 0, 0), 2, CV_AA);

		cv::imwrite(save_dir + "final_2D.jpg", overlay);
	}
}

void Fusion::lidar_to_cam(const Eigen::Vector3f &lidar_point, Eigen::Vector3f &cam_point)
{
	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);

	X.at<double>(0, 0) = lidar_point(0);
	X.at<double>(1, 0) = lidar_point(1);
	X.at<double>(2, 0) = lidar_point(2);
	X.at<double>(3, 0) = 1;

	Y = Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

	cam_point(0) = Y.at<double>(0, 0);
	cam_point(1) = Y.at<double>(1, 0);
	cam_point(2) = Y.at<double>(2, 0);

	// std::cout << "x: " << cam_point(0) << " y: " << cam_point(1) << " z: " << cam_point(2) << std::endl;
}

void Fusion::draw_projected_box3d(const cv::Mat &raw_image, const std::vector<pcl::PointCloud<pcl::PointXYZ>> boxes, const std::vector<std::pair<int, int>> matchedPairs)
{
	/*
		Draw 3d bounding box in image
		qs: (8,3) array of vertices for the 3d box in following order:
		 1 --------- 0
		/|          /|
		2 -------- 3 |
		| |        | |
		| 5 -------| 4
		|/         |/
		6 -------- 7
	*/

	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);
	cv::Mat visImg = raw_image.clone();
	cv::Point pt_start;
	cv::Point pt_end;

	for (const auto &match : matchedPairs)
	{
		int i = match.first;
		X.at<double>(0, 0) = boxes[i][0].x;
		X.at<double>(1, 0) = boxes[i][0].y;
		X.at<double>(2, 0) = boxes[i][0].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][1].x;
		X.at<double>(1, 0) = boxes[i][1].y;
		X.at<double>(2, 0) = boxes[i][1].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][1].x;
		X.at<double>(1, 0) = boxes[i][1].y;
		X.at<double>(2, 0) = boxes[i][1].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][2].x;
		X.at<double>(1, 0) = boxes[i][2].y;
		X.at<double>(2, 0) = boxes[i][2].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][2].x;
		X.at<double>(1, 0) = boxes[i][2].y;
		X.at<double>(2, 0) = boxes[i][2].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][3].x;
		X.at<double>(1, 0) = boxes[i][3].y;
		X.at<double>(2, 0) = boxes[i][3].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][3].x;
		X.at<double>(1, 0) = boxes[i][3].y;
		X.at<double>(2, 0) = boxes[i][3].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][0].x;
		X.at<double>(1, 0) = boxes[i][0].y;
		X.at<double>(2, 0) = boxes[i][0].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][4].x;
		X.at<double>(1, 0) = boxes[i][4].y;
		X.at<double>(2, 0) = boxes[i][4].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][5].x;
		X.at<double>(1, 0) = boxes[i][5].y;
		X.at<double>(2, 0) = boxes[i][5].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][5].x;
		X.at<double>(1, 0) = boxes[i][5].y;
		X.at<double>(2, 0) = boxes[i][5].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][6].x;
		X.at<double>(1, 0) = boxes[i][6].y;
		X.at<double>(2, 0) = boxes[i][6].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][6].x;
		X.at<double>(1, 0) = boxes[i][6].y;
		X.at<double>(2, 0) = boxes[i][6].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][7].x;
		X.at<double>(1, 0) = boxes[i][7].y;
		X.at<double>(2, 0) = boxes[i][7].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][7].x;
		X.at<double>(1, 0) = boxes[i][7].y;
		X.at<double>(2, 0) = boxes[i][7].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][4].x;
		X.at<double>(1, 0) = boxes[i][4].y;
		X.at<double>(2, 0) = boxes[i][4].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][0].x;
		X.at<double>(1, 0) = boxes[i][0].y;
		X.at<double>(2, 0) = boxes[i][0].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][4].x;
		X.at<double>(1, 0) = boxes[i][4].y;
		X.at<double>(2, 0) = boxes[i][4].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][1].x;
		X.at<double>(1, 0) = boxes[i][1].y;
		X.at<double>(2, 0) = boxes[i][1].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][5].x;
		X.at<double>(1, 0) = boxes[i][5].y;
		X.at<double>(2, 0) = boxes[i][5].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][2].x;
		X.at<double>(1, 0) = boxes[i][2].y;
		X.at<double>(2, 0) = boxes[i][2].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][6].x;
		X.at<double>(1, 0) = boxes[i][6].y;
		X.at<double>(2, 0) = boxes[i][6].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);

		X.at<double>(0, 0) = boxes[i][3].x;
		X.at<double>(1, 0) = boxes[i][3].y;
		X.at<double>(2, 0) = boxes[i][3].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_start.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_start.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		X.at<double>(0, 0) = boxes[i][7].x;
		X.at<double>(1, 0) = boxes[i][7].y;
		X.at<double>(2, 0) = boxes[i][7].z;
		X.at<double>(3, 0) = 1;

		Y = Calib_P2 * Calib_R_rect * Calib_Tr_velo_to_cam * X; // tranform the point to the camera coordinate

		pt_end.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
		pt_end.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

		cv::line(visImg, cv::Point(pt_start.x, pt_start.y), cv::Point(pt_end.x, pt_end.y), cv::Scalar(255, 0, 0), 2, CV_AA);
	}

	cv::imwrite(save_dir + "3D_img.jpg", visImg);
}
