#pragma once

#include <iostream>
#include <vector>
#include <string.h>
#include <Eigen/Dense>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

extern Eigen::Matrix<double, 3, 4> P2;
extern Eigen::Matrix<double, 4, 4> R_rect;
extern Eigen::Matrix<double, 4, 4> Tr_velo_to_cam;

typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> PointICloud;
typedef PointICloud::Ptr PointICloudPtr;
typedef PointICloud::ConstPtr PointICloudConstPtr;

void load_Calibration(std::string file_name);

void fov_segmentation(PointICloudPtr &cloudXYZI, PointICloudPtr &cloud_fov);

void load_bin_cloud(std::string kitti_filename, pcl::PointCloud<pcl::PointXYZI>::Ptr &point_cloud);

bool endsWith(const std::string& str, const std::string& suffix);

std::string removeExtension(const std::string& fileName);

std::vector<std::string> getTxtFileNames(const std::string& folderPath);