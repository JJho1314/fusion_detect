#ifndef BOX_H
#define BOX_H
#include <Eigen/Geometry>

#define OBJ_NAME_MAX_SIZE 16

struct BoxQ
{
    Eigen::Vector3f bboxTransform;
    Eigen::Quaternionf bboxQuaternion;
    float cube_length;
    float cube_width;
    float cube_height;
};
struct BBox3D
{
    float x_min;
    float y_min;
    float z_min;
    float x_max;
    float y_max;
    float z_max;
};

struct BBox2D
{
    float x_min;
    float y_min;
    float x_max;
    float y_max;

    float prop;
    char name[OBJ_NAME_MAX_SIZE];
};

#endif