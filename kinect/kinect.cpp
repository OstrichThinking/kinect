#include<iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <kinect.h>

using namespace std;
using namespace cv;

// to release the pointer safely 
template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease)
{
    if (pInterfaceToRelease != NULL)
    {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}


int main(int argc, char* argv[])
{
    // get inect sensor
    IKinectSensor* kinectSensor = nullptr;
    HRESULT RESULT;
    
    RESULT = GetDefaultKinectSensor(&kinectSensor);
    if (FAILED(RESULT)){
        return RESULT;
    }

    IMultiSourceFrameReader* multiFrameReader = nullptr;
    if (kinectSensor){
        RESULT = kinectSensor->Open();
        if (SUCCEEDED(RESULT)){
            RESULT = kinectSensor->OpenMultiSourceFrameReader(
                FrameSourceTypes::FrameSourceTypes_Color |
                FrameSourceTypes::FrameSourceTypes_Infrared |
                FrameSourceTypes::FrameSourceTypes_Depth,
                &multiFrameReader);
        }
    }

    IDepthFrameReference* depthFrameReference = nullptr;
    IColorFrameReference* colorFrameReference = nullptr;
    IInfraredFrameReference* infraredFrameReference = nullptr;

    IInfraredFrame* infraredFrame = nullptr;
    IDepthFrame* depthFrame = nullptr;
    IColorFrame* colorFrame = nullptr;

    Mat i_rgb(1080, 1920, CV_8UC4);
    Mat i_depth(424, 512, CV_16UC1);
    Mat i_ir(424, 512, CV_16UC1);

    IMultiSourceFrame* multiFrame = nullptr;

    while (true)
    {
        // get a latest multiFrame
        RESULT = multiFrameReader->AcquireLatestFrame(&multiFrame);
        
        // get deepth infrared color frame from multiFrame
        if (SUCCEEDED(RESULT))
            RESULT = multiFrame->get_ColorFrameReference(&colorFrameReference);
        if (SUCCEEDED(RESULT))
            RESULT = colorFrameReference->AcquireFrame(&colorFrame);
        if (SUCCEEDED(RESULT))
            RESULT = multiFrame->get_DepthFrameReference(&depthFrameReference);
        if (SUCCEEDED(RESULT))
            RESULT = depthFrameReference->AcquireFrame(&depthFrame);
        if (SUCCEEDED(RESULT))
            RESULT = multiFrame->get_InfraredFrameReference(&infraredFrameReference);
        if (SUCCEEDED(RESULT))
            RESULT = infraredFrameReference->AcquireFrame(&infraredFrame);

        // color frame to mat 1920*1080*4
        UINT nColorBufferSize = 1920 * 1080 * 4;
        if (SUCCEEDED(RESULT)) {
            RESULT = colorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(i_rgb.data), ColorImageFormat::ColorImageFormat_Bgra);
            
            imshow("rgb", i_rgb);
            if (waitKey(1) == VK_ESCAPE)
                break;
        }

        // depth frame to mat 424*512*1
        if (SUCCEEDED(RESULT)){
            RESULT = depthFrame->CopyFrameDataToArray(424 * 512, reinterpret_cast<UINT16*>(i_depth.data));

            //16 to 8
            int width = i_depth.cols;
            int height = i_depth.rows;
            Mat dst_8 = Mat::zeros(height, width, CV_8UC1);//create a enpty mat
            double minv = 0.0, maxv = 0.0;
            double* minp = &minv;
            double* maxp = &maxv;
            minMaxIdx(i_depth, minp, maxp);  //get the max and min of pixel

            //cout << "min of pixel:  " << minv << "  max of piexl:  " << maxv << endl;

            ushort* p_img;
            uchar* p_dst;
            for (int i = 0; i < height; i++){

                p_img = i_depth.ptr<ushort>(i);//get i row head pointer in source_img
                p_dst = dst_8.ptr<uchar>(i);//get i row head pointer in des_img
                for (int j = 0; j < width; ++j)
                {
                    p_dst[j] = (p_img[j] - minv) / (maxv - minv) * 255;
                }
            }

            // deepth to color  COLORMAP_JET
            Mat color;
            applyColorMap(dst_8, color, COLORMAP_JET);


            imshow("depth", color);
            if (waitKey(1) == VK_ESCAPE)
                break;
        }

        // infrared frame to mat 424*512*1
        if (SUCCEEDED(RESULT)) {
            RESULT = infraredFrame->CopyFrameDataToArray(424 * 512, reinterpret_cast<UINT16*>(i_ir.data));

            imshow("infrared", i_ir);
            if (waitKey(1) == VK_ESCAPE)
                break;
        }

        // release resourse
        SafeRelease(colorFrame);
        SafeRelease(depthFrame);
        SafeRelease(infraredFrame);
        SafeRelease(colorFrameReference);
        SafeRelease(depthFrameReference);
        SafeRelease(infraredFrameReference);
        SafeRelease(multiFrame);
    }

    return 0;
}
