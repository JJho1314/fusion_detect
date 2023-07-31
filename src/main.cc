/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
// #include <ncurses.h>
#include <dirent.h>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
// #include <ncurses.h>

#define _BASETSD_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#undef cimg_display
#define cimg_display 0
#undef cimg_use_jpeg
#define cimg_use_jpeg 1
#undef cimg_use_png
#define cimg_use_png 1
#include "CImg/CImg.h"

#include <pcl/io/pcd_io.h>   //PCL的PCD格式文件的输入输出头文件
#include <pcl/point_types.h> //PCL对各种格式的点的支持头文件
// #include "drm_func.h"
// #include "rga_func.h"
#include "rknn_api.h"
#include "yolo.h"
#include "common.h"
#include "tracker.h"
#include "fusion.hpp"
#include "common.hpp"
#include "box_fitting.h"
#include "CVC_cluster.h"
#include "patchwork.hpp"
#include "GuassianProcess.h"

// #define PLATFORM_RK3588
#define PERF_WITH_POST 0
#define COCO_IMG_NUMBER 5000
#define DUMP_INPUT 0

using namespace cimg_library;
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
           attr->dims[2], attr->dims[3], attr->n_elems, attr->size, 0, attr->type,
           attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int query_model_info(MODEL_INFO *m, rknn_context ctx)
{
    int ret;
    /* Query sdk version */
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    /* Get input,output attr */
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);
    m->in_nodes = io_num.n_input;
    m->out_nodes = io_num.n_output;
    m->in_attr = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
    m->out_attr = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
    if (m->in_attr == NULL || m->out_attr == NULL)
    {
        printf("alloc memery failed\n");
        return -1;
    }

    for (int i = 0; i < io_num.n_input; i++)
    {
        m->in_attr[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &m->in_attr[i],
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&m->in_attr[i]);
    }

    for (int i = 0; i < io_num.n_output; i++)
    {
        m->out_attr[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(m->out_attr[i]),
                         sizeof(rknn_tensor_attr));
        printRKNNTensor(&(m->out_attr[i]));
    }

    /* get input shape */
    if (io_num.n_input > 1)
    {
        printf("expect model have 1 input, but got %d\n", io_num.n_input);
        return -1;
    }

    if (m->in_attr[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        m->width = m->in_attr[0].dims[0];
        m->height = m->in_attr[0].dims[1];
        m->channel = m->in_attr[0].dims[2];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        m->width = m->in_attr[0].dims[2];
        m->height = m->in_attr[0].dims[1];
        m->channel = m->in_attr[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", m->height, m->width,
           m->channel);

    return 0;
}

static int status = 0;
static rknn_context ctx;
unsigned int handle;
MODEL_INFO m_info;
LETTER_BOX letter_box;
static size_t actual_size = 0;
static int img_width = 0;
static int img_height = 0;
static int img_channel = 0;
void *resize_buf;
unsigned char *model_data;
static int startX, startY;
static float img_ratio;
std::string model_path;
std::string root_dir;
std::string LABEL_NALE_TXT_PATH;
int OBJ_CLASS_NUM;
int PROP_BOX_SIZE;

static struct timeval start_time, stop_time;
static int ret;

std::string save_dir;

int DetectorInit(MODEL_INFO *m)
{
    int status = 0;

    m->m_type = YOLOX;
    m->anchor_per_branch = 1;
    m->post_type = Q8;

    const char *charPtr = model_path.c_str();
    m->m_path = strdup(charPtr);
    char *anchor_path = " ";
    // 输入图像地址
    m->in_path = "";
    for (int i = 0; i < 18; i++)
    {
        m->anchors[i] = 1;
    }
    if (ret < 0)
        return -1;

    /* Create the neural network */
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(m_info.m_path, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    printf("query info\n");
    ret = query_model_info(&m_info, ctx);
    if (ret < 0)
    {
        return -1;
    }
}

cv::Mat preprocess(const cv::Mat originalImage)
{
    int originalWidth = originalImage.cols;
    int originalHeight = originalImage.rows;
    // 创建一个新的 640x640 大小的黑色图像
    cv::Mat resizedImage(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    // 计算调整大小后的图像的宽度和高度
    int resizedWidth, resizedHeight;
    if (originalWidth > originalHeight)
    {
        resizedWidth = 640;
        img_ratio = originalWidth / 640.0;
        resizedHeight = originalHeight * 640 / originalWidth;
    }
    else
    {
        resizedWidth = originalWidth * 640 / originalHeight;
        img_ratio = originalHeight / 640.0;
        resizedHeight = 640;
    }
    // 计算调整大小后图像的起始坐标
    startX = (640 - resizedWidth) / 2;
    startY = (640 - resizedHeight) / 2;

    // 调整大小并将原始图像复制到新图像中
    cv::resize(originalImage, resizedImage(cv::Rect(startX, startY, resizedWidth, resizedHeight)), cv::Size(resizedWidth, resizedHeight));
    return resizedImage;
}

void analysisYaml()
{
    std::string filePath = "../config.yaml";
    try
    {
        // 加载YAML文件
        YAML::Node config = YAML::LoadFile(filePath);

        // 读取person节点中的数据
        model_path = config["model_path"].as<std::string>();

        LABEL_NALE_TXT_PATH = config["LABEL_NALE_TXT_PATH"].as<std::string>();
        OBJ_CLASS_NUM = config["OBJ_CLASS_NUM"].as<int>();
        PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM;
    }
    catch (const YAML::Exception &e)
    {
        std::cout << "Failed to load YAML file: " << e.what() << std::endl;
    }
}

int DetectorRun(const cv::Mat &img, std::vector<BBox2D> &img_boxes2D)
{
    /* Init input tensor */
    rknn_input inputs[1];

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8; /* SAME AS INPUT IMAGE */
    inputs[0].size = m_info.width * m_info.height * m_info.channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC; /* SAME AS INPUT IMAGE */
    inputs[0].pass_through = 0;
    // std::cout << "m_info:whc" << m_info.width << " " << m_info.height << " " << m_info.channel;

    /* Init output tensor */
    rknn_output outputs[m_info.out_nodes];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < m_info.out_nodes; i++)
    {
        // printf("The info type: %d\n", m_info.post_type);
        outputs[i].want_float = m_info.post_type;
    }
    void *resize_buf = malloc(inputs[0].size);
    if (resize_buf == NULL)
    {
        printf("resize buf alloc failed\n");
        return -1;
    }
    void *rk_outputs_buf[m_info.out_nodes];

    cv::Mat pre_img = preprocess(img);

    inputs[0].buf = pre_img.data;
    // gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, m_info.in_nodes, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, m_info.out_nodes, outputs, NULL);

    /* Post process */
    detect_result_group_t detect_result_group;
    for (auto i = 0; i < m_info.out_nodes; i++)
        rk_outputs_buf[i] = outputs[i].buf;
    post_process(rk_outputs_buf, &m_info, &detect_result_group, LABEL_NALE_TXT_PATH, img_ratio, startX, startY);

    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n",
           (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // Draw Objects
    const unsigned char blue[] = {0, 0, 255};
    char score_result[64];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        BBox2D img_box2D;
        char text[256];

        printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom, det_result->prop);

        // draw box
        std::string name(det_result->name);

        if (det_result->prop > 0.4 && (name == "Pedestrian" || name == "Car"))
        {
            sprintf(text, "%s", det_result->name);

            img_box2D.x_min = det_result->box.left;
            img_box2D.y_min = det_result->box.top;
            img_box2D.x_max = det_result->box.right;
            img_box2D.y_max = det_result->box.bottom;
            img_box2D.prop = det_result->prop;
            strcpy(img_box2D.name, det_result->name);
            // img_box2D.name = det_result->name;

            img_boxes2D.emplace_back(img_box2D);

            cv::rectangle(img, cv::Point(img_box2D.x_min, img_box2D.y_min), cv::Point(img_box2D.x_max, img_box2D.y_max), cv::Scalar(147, 20, 255), 2);
            putText(img, text, cv::Point(img_box2D.x_min, img_box2D.y_min), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 3);
        }
    }

    // img.save("./out.bmp");
    cv::imwrite(save_dir + "out.jpg", img);
    ret = rknn_outputs_release(ctx, m_info.out_nodes, outputs);
}

void DetectorRelease()
{
    // release
    ret = rknn_destroy(ctx);

    if (model_data)
    {
        free(model_data);
    }

    if (m_info.in_attr)
    {
        free(m_info.in_attr);
    }

    if (m_info.out_attr)
    {
        free(m_info.out_attr);
    }
}

using PointType = pcl::PointXYZI;
boost::shared_ptr<PatchWork<PointType>> PatchworkGroundSeg;
int main(int argc, char **argv)
{
    analysisYaml();
    DetectorInit(&m_info);

    gettimeofday(&start_time, NULL);

    // 读取图像文件
    root_dir = argv[1];
    std::string velodyne_dir = root_dir + "velodyne/";
    // std::string label_dir = root_dir + "label_2/";
    std::string calib_dir = root_dir + "calib/";
    std::string image_dir = root_dir + "image/";

    std::string result_dir = "../results/";
    std::string save_data_dir = "data/";

    std::vector<std::string> FileNames;

    FileNames = getTxtFileNames(calib_dir);

    std::sort(FileNames.begin(), FileNames.end());

    std::string num_idex;

    for (int i = 0; i < FileNames.size(); i++)
    {
        num_idex = FileNames[i];
        // num_idex = "000089";

        std::string kitti_cloud_filename = velodyne_dir + num_idex + ".bin";
        // std::string kitti_label_filename = label_dir + num_idex + ".txt";
        std::string kitti_calib_filename = calib_dir + num_idex + ".txt";
        std::string kitti_image_filename = image_dir + num_idex + ".png";

        std::cout << kitti_calib_filename << std::endl;

        const char *image_name = kitti_image_filename.c_str();

        save_dir = result_dir + num_idex + "/";
        mkdir(result_dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        mkdir((result_dir + num_idex).c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        mkdir((result_dir + save_data_dir).c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);

        cv::Mat image = cv::imread(image_name);

        cv::Mat orig_img = image.clone();

        /****************************************  初始化点云  *************************************/

        // load point cloud
        PointICloudPtr point_cloud(new PointICloud);
        PointICloudPtr cloud_fov(new PointICloud);

        load_bin_cloud(kitti_cloud_filename, point_cloud);
        load_Calibration(kitti_calib_filename);

        // 获取视野范围内的点云
        fov_segmentation(point_cloud, cloud_fov);

        /****************************************  点云处理  ******************************************/

        pcl::PointCloud<PointType>::Ptr pc_curr(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pc_ground(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pc_non_ground(new pcl::PointCloud<PointType>);
        double time_taken;

        // /***************************** ***** Patchwork的地面分割  **********************************/

        // PatchworkGroundSeg.reset(new PatchWork<PointType>());

        // for (int i = 0; i < cloud_fov->points.size(); ++i)
        // {
        //     PointType p;
        //     p.x = cloud_fov->points[i].x;
        //     p.y = cloud_fov->points[i].y;
        //     p.z = cloud_fov->points[i].z;
        //     p.intensity = cloud_fov->points[i].intensity;
        //     pc_curr->points.push_back(p);
        // }

        // PatchworkGroundSeg->estimate_ground(*pc_curr, *pc_ground, *pc_non_ground, time_taken);

        /*****************************************    GP INSAC    ************************************/
        std::cout << "GP_INSAC" << std::endl;

        autosense::segmenter::GP_INSAC(*cloud_fov, *pc_non_ground, *pc_ground);

        /****************    CVC聚类    **************************/

        vector<float> param(3, 0);
        param[0] = 3.0;
        param[1] = 0.5;
        param[2] = 0.8;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_point(new pcl::PointCloud<pcl::PointXYZ>);

        for (int i = 0; i < pc_non_ground->points.size(); ++i)
        {
            pcl::PointXYZ p;
            p.x = pc_non_ground->points[i].x;
            p.y = pc_non_ground->points[i].y;
            p.z = pc_non_ground->points[i].z;
            cluster_point->points.push_back(p);
        }

        CVC Cluster(param);
        std::vector<PointAPR> capr;
        Cluster.calculateAPR(*cluster_point, capr);

        std::unordered_map<int, Voxel> hash_table;
        Cluster.build_hash_table(capr, hash_table);
        vector<int> cluster_indices;
        cluster_indices = Cluster.cluster(hash_table, capr);
        vector<int> cluster_id;
        Cluster.most_frequent_value(cluster_indices, cluster_id);

        std::vector<pcl::PointCloud<pcl::PointXYZ>> cloudClusters; // 所有聚类的集合
        std::vector<BBox3D> boxes3D;                               // 所有3D包围框集合

        for (int j = 0; j < cluster_id.size(); ++j)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudcluster(new pcl::PointCloud<pcl::PointXYZ>); // 初始化
            for (int i = 0; i < cluster_indices.size(); ++i)
            {
                if (cluster_indices[i] == cluster_id[j])
                {
                    cloudcluster->points.emplace_back(cluster_point->points[i]);
                }
            }
            cloudClusters.emplace_back(*cloudcluster);
        }
        std::cout << "聚类数量： " << cloudClusters.size() << std::endl;

        std::vector<pcl::PointCloud<pcl::PointXYZ>> bbPoints; // 出来的是所有框的八个顶点

        std::vector<BoxQ> boxesQ; // 出来的是所有框的中心点，边长，朝向

        Box_fitting box_fitting;

        box_fitting.getBoundingBox(cloudClusters, bbPoints, true); // 生成最小包围矩形框输出为包围框的八个顶点  // TODO根据距离和车人的先验去过滤

        if (bbPoints.size() <= 0) // 框都没有就不过滤了
        {
            box_fitting.getBoundingBox(cloudClusters, bbPoints, false);
        }

        box_fitting.calculateBoundingBoxes(bbPoints, boxesQ); // 将八个顶点转换成所有框的中心点，边长，朝向

        /********************************************   目标检测   *************************************************/

        std::vector<BBox2D> img_boxes2D;
        DetectorRun(image, img_boxes2D);

        /*************************************    雷达3D框转到图像2D框     ******************************************/

        Fusion fusion_point_img;
        std::vector<BBox2D> lidar_boxes2D;

        fusion_point_img.convert_3DBox_to_2DBox(orig_img, bbPoints, lidar_boxes2D); // 将投影的3D框转换成最大包围的2D框

        /***********************************   3D-2D包围框融合   ************************************/

        auto [matchedPairs, unmatchedLidarBoxes, unmatchedCameraBoxes] = fusion_point_img.associate(boxesQ, lidar_boxes2D, img_boxes2D); // TODO 根据聚类最近点距离调整交并比阈值

        std::cout << "匹配成功的数量： " << matchedPairs.size() << std::endl;

        /*************************************  融合点云到图像  *************************************/
        cv::Mat final_fusion_img = cv::imread(save_dir + "3D_to_2D_img.jpg");

        std::string img_front = "/lx3_";

        if (atoi(num_idex.c_str()) < 10)
        {
            num_idex = "00" + std::to_string(atoi(num_idex.c_str()));
        }
        else if (atoi(num_idex.c_str()) >= 10 && (atoi(num_idex.c_str()) < 100))
        {
            num_idex = "0" + std::to_string(atoi(num_idex.c_str()));
        }
        else if (atoi(num_idex.c_str()) >= 100)
        {
            num_idex = std::to_string(atoi(num_idex.c_str()));
        }

        std::fstream f;
        f.open(result_dir + save_data_dir + img_front + num_idex + ".txt", ios::out);

        fusion_point_img.pointcloud2_to_image(orig_img, point_cloud); // 融合全部点云到图片

        if (matchedPairs.size() <= 0)
        {
            continue;
        }

        fusion_point_img.draw_projected_box3d(orig_img, bbPoints, matchedPairs); // 将3D框画到图像上

        cv::Mat img_3D = cv::imread(save_dir + "3D_img.jpg");

        fusion_point_img.cluster_to_image(img_3D, point_cloud, matchedPairs, cloudClusters); // 融合聚类点云到图片上

        fusion_point_img.build_fused_object(orig_img, point_cloud, lidar_boxes2D, matchedPairs, cloudClusters); // 建立3D包围框的融合关系

        /*******************************************************************************************/

        /*************************************  融合点云到图像  *************************************/
        cv::Mat final_2D = cv::imread(save_dir + "final_2D.jpg");

        cv::Mat final_3D = cv::imread(save_dir + "cluster_to_img.jpg");

        // Print the matched pairs
        for (const auto &match : matchedPairs)
        {
            char text[256];
            std::cout << "LiDAR box index: " << match.first << ", Camera box index: " << match.second << std::endl;

            Eigen::Vector3f center = boxesQ[match.first].bboxTransform;

            Eigen::Vector3f cam_point;
            fusion_point_img.lidar_to_cam(center, cam_point);

            // double rotation_y = atan2(cam_point_1(2) - cam_point_2(2), cam_point_1(0) - cam_point_2(0));
            Eigen::Matrix3f lidar_rotation_matrix;
            Eigen::Matrix3f camera_rotation_matrix;
            lidar_rotation_matrix = boxesQ[match.first].bboxQuaternion.toRotationMatrix();

            camera_rotation_matrix = Tr_velo_to_cam.block(0, 0, 3, 3).cast<float>() * lidar_rotation_matrix;

            // 旋转矩阵to欧拉角 ZYX顺序，0表示X轴,1表示Y轴,2表示Z轴
            Eigen::Vector3f euler_angles = camera_rotation_matrix.eulerAngles(2, 1, 0); // z-y-x

            float rotation_y = euler_angles(1); // y

            // 输入你想写入的内容 对于kitti要注意雷达坐标系和相机坐标系的不同
            f << img_boxes2D[match.second].name << " "
              << "-1"
              << " "
              << "-1"
              << " "
              << "-1"
              << " ";
            f << "-1"
              << " "
              << "-1"
              << " "
              << "-1"
              << " "
              << "-1"
              << " ";
            f << std::to_string(boxesQ[match.first].cube_height) << " " << std::to_string(boxesQ[match.first].cube_width) << " " << std::to_string(boxesQ[match.first].cube_length) << " ";
            f << std::to_string(cam_point(0)) << " " << std::to_string(cam_point(1)) << " " << std::to_string(cam_point(2)) << " " << std::to_string(rotation_y) << " " << std::to_string(img_boxes2D[match.second].prop);
            f << endl;

            sprintf(text, "%s", img_boxes2D[match.second].name);
            cv::rectangle(final_2D, cv::Point(img_boxes2D[match.second].x_min, img_boxes2D[match.second].y_min), cv::Point(img_boxes2D[match.second].x_max, img_boxes2D[match.second].y_max), cv::Scalar(147, 20, 255), 2);
            putText(final_2D, text, cv::Point(img_boxes2D[match.second].x_min, img_boxes2D[match.second].y_min), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 3);
            // cv::rectangle(final_3D, cv::Point(img_boxes2D[match.second].x_min, img_boxes2D[match.second].y_min), cv::Point(img_boxes2D[match.second].x_max, img_boxes2D[match.second].y_max), cv::Scalar(147, 20, 255), 2);
            // putText(final_3D, text, cv::Point(img_boxes2D[match.second].x_min, img_boxes2D[match.second].y_min), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 3);
        }

        f.close();

        // 保存图像文件到目标文件夹
        cv::imwrite(save_dir + "result.jpg", final_2D);
        cv::imwrite(save_dir + "final_3D.jpg", final_3D);
    }

    gettimeofday(&stop_time, NULL);
    printf("single frame run use %f ms\n",
           (__get_us(stop_time) - __get_us(start_time)) / 1000);

    DetectorRelease();
}
