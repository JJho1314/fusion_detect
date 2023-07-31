#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <dirent.h>

#include <Eigen/Dense>
#include <unistd.h>

#include "common.hpp"

// 1.释放内存 *data 2.保存point_cloud为PCD，注意设定长和高
void load_bin_cloud(std::string kitti_filename, pcl::PointCloud<pcl::PointXYZI>::Ptr &point_cloud)
{
    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    int32_t num = 1000000;
    float *data = (float *)malloc(num * sizeof(float)); // void *malloc(size_t size) 分配所需的内存空间，并返回一个指向它的指针。

    // pointers
    float *px = data + 0;
    float *py = data + 1;
    float *pz = data + 2;
    float *pr = data + 3;

    FILE *stream;
    stream = fopen(kitti_filename.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream) / 4;
    point_cloud->width = num;      // 设定长
    point_cloud->height = 1;       // 设定高
    point_cloud->is_dense = false; // 如果没有无效点（例如，具有NaN或Inf值），则为True
    for (int32_t i = 0; i < num; i++)
    {
        // vector<int32_t> point_cloud;
        pcl::PointXYZI point;
        point.x = *px;
        point.y = *py;
        point.z = *pz;
        point.intensity = *pr;
        point_cloud->points.push_back(point);
        px += 4;
        py += 4;
        pz += 4;
        pr += 4;
    }
    fclose(stream);
    free(data); // 释放内存
}

Eigen::Matrix<double, 3, 4> P2;
Eigen::Matrix<double, 4, 4> R_rect;
Eigen::Matrix<double, 4, 4> Tr_velo_to_cam;
void load_Calibration(std::string file_name)
{
    FILE *fp = fopen(file_name.c_str(), "r");
    if (!fp)
    {
        printf("open Calib error!!!\n");
        return;
    }
    char str[255];
    double temp[12];

    // P0 && p1
    for (int i = 0; i < 2; i++)
    {
        fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               str, &temp[0], &temp[1], &temp[2], &temp[3], &temp[4], &temp[5], &temp[6],
               &temp[7], &temp[8], &temp[9], &temp[10], &temp[11]);
    }
    // p2
    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           str, &temp[0], &temp[1], &temp[2], &temp[3], &temp[4], &temp[5], &temp[6],
           &temp[7], &temp[8], &temp[9], &temp[10], &temp[11]);

    P2 << temp[0], temp[1], temp[2], temp[3],
        temp[4], temp[5], temp[6], temp[7],
        temp[8], temp[9], temp[10], temp[11];
    // p3
    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           str, &temp[0], &temp[1], &temp[2], &temp[3], &temp[4], &temp[5], &temp[6],
           &temp[7], &temp[8], &temp[9], &temp[10], &temp[11]);

    /// R0_rect
    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           str, &temp[0], &temp[1], &temp[2], &temp[3], &temp[4], &temp[5], &temp[6],
           &temp[7], &temp[8]);

    R_rect << temp[0], temp[1], temp[2], 0,
        temp[3], temp[4], temp[5], 0,
        temp[6], temp[7], temp[8], 0,
        0, 0, 0, 1;

    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           str, &temp[0], &temp[1], &temp[2], &temp[3], &temp[4], &temp[5], &temp[6],
           &temp[7], &temp[8], &temp[9], &temp[10], &temp[11]);

    Tr_velo_to_cam << temp[0], temp[1], temp[2], temp[3],
        temp[4], temp[5], temp[6], temp[7],
        temp[8], temp[9], temp[10], temp[11],
        0, 0, 0, 1;

    fclose(fp);
}

bool endsWith(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string removeExtension(const std::string &fileName)
{
    size_t lastDotIndex = fileName.find_last_of(".");
    if (lastDotIndex != std::string::npos)
    {
        return fileName.substr(0, lastDotIndex);
    }
    return fileName;
}

std::vector<std::string> getTxtFileNames(const std::string &folderPath)
{
    std::vector<std::string> txtFileNames;
    DIR *directory = opendir(folderPath.c_str());

    if (directory == nullptr)
    {
        std::cout << "Invalid folder path or folder does not exist." << std::endl;
        return txtFileNames;
    }

    struct dirent *entry;
    while ((entry = readdir(directory)) != nullptr)
    {
        std::string fileName = entry->d_name;
        if (endsWith(fileName, ".txt"))
        {
            txtFileNames.push_back(removeExtension(fileName));
        }
    }

    closedir(directory);
    return txtFileNames;
}

const int UMax = 1242;
const int VMax = 375;
void fov_segmentation(PointICloudPtr &cloudXYZI, PointICloudPtr &cloud_fov)
{
    /****************************************************/
    //=============     只在KITTI检测范围内的点云    =============//

    for (int i = 0; i < cloudXYZI->size(); ++i)
    {
        // lidar_to_rect
        if ((*cloudXYZI)[i].x < 0 || (*cloudXYZI)[i].x > 70 || std::abs((*cloudXYZI)[i].y) > 40)
            continue;
        Eigen::Matrix<double, 4, 1> center;
        Eigen::Matrix<double, 4, 1> center_3D;
        center_3D << (*cloudXYZI)[i].x, (*cloudXYZI)[i].y, (*cloudXYZI)[i].z, 1;
        center = R_rect * Tr_velo_to_cam * center_3D;
        // rect_to_img
        Eigen::Matrix<double, 3, 1> pts_2d_hom = P2 * center;
        pts_2d_hom(0, 0) /= pts_2d_hom(2, 0);
        pts_2d_hom(1, 0) /= pts_2d_hom(2, 0);
        double pts_rect_depth = pts_2d_hom(2, 0) - P2(2, 3);

        // get_fov_flag
        if (pts_2d_hom(0, 0) >= 0 && pts_2d_hom(0, 0) < UMax &&
            pts_2d_hom(1, 0) >= 0 && pts_2d_hom(1, 0) < VMax &&
            pts_rect_depth >= 0)
        {
            PointI point = (*cloudXYZI)[i];
            cloud_fov->push_back(point);
        }
    }
    cloudXYZI = cloud_fov;
}
