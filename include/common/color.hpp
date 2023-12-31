/*
 * Copyright (C) 2019 by AutoSense Organization. All rights reserved.
 * Gary Chan <chenshj35@mail2.sysu.edu.cn>
 */

#ifndef COMMON_INCLUDE_COMMON_COLOR_HPP_
#define COMMON_INCLUDE_COMMON_COLOR_HPP_

#include <std_msgs/ColorRGBA.h>  // std_msgs::ColorRGBA

namespace autosense {
namespace common {
//----------------------------------- color utils
/**
 * Black           0   0    0    0
   White           255    255    255    16777215
   Gray            192    192    192    12632256
   Dark Grey       128    128    128    8421504
   Red             255    0    0    255
   Dark Red        128    0    0    128
   Green           0    255    0    65280
   Dark Green      0    128    0    32768
   Blue            0    0    255    16711680
   Dark Blue       0    0    128    8388608
   Magenta         255    0    255    16711935
   Dark Magenta    128    0    128    8388736
   Cyan            0    255    255    16776960
   Dark Cyan       0    128    128    8421376
   Yellow          255    255    0    65535
   Brown           128    128    0    32896
 */
struct Color {
    Color(float r, float g, float b) {
        rgbA.r = r;
        rgbA.g = g;
        rgbA.b = b;
        rgbA.a = 1.0;
    }

    std_msgs::ColorRGBA rgbA;
};

const struct Color BLACK(0.0, 0.0, 0.0);
const struct Color WHITE(1.0, 1.0, 1.0);
const struct Color RED(1.0, 0.0, 0.0);
const struct Color DARKRED(0.5, 0.0, 0.0);
const struct Color GREEN(0.0, 1.0, 0.0);
const struct Color DARKGREEN(0.0, 0.5, 0.0);
const struct Color BLUE(0.0, 0.0, 1.0);
const struct Color DARKBLUE(0.0, 0.0, 0.5);
const struct Color MAGENTA(1.0, 0.0, 1.0);
const struct Color DARKMAGENTA(0.5, 0.0, 0.5);
const struct Color CYAN(0.0, 1.0, 1.0);
const struct Color DARKCYAN(0.0, 0.5, 0.5);
const struct Color YELLOW(1.0, 1.0, 0.0);
const struct Color BROWN(0.5, 0.5, 0.0);
const struct Color NEWBLUE(0.18, 0.45, 0.70);

}  // namespace common
}  // namespace autosense

#endif  // COMMON_INCLUDE_COMMON_COLOR_HPP_
