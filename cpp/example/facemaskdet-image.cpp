/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2020, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facemaskdetcnn.h"

// define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

string labelmap[3] = {"__background__", "face", "mask"}; 

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        printf("Usage: %s <image_file_name>\n", argv[0]);
        return -1;
    }

	//load an image and convert it to gray (single-channel)
	Mat image = imread(argv[1]); 
	if(image.empty())
	{
		fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
		return -1;
	}

	int * pResults = NULL; 
    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }
	

	///////////////////////////////////////////
	// CNN face detection 
	// Best detection rate
	//////////////////////////////////////////
	//!!! The input image must be a BGR one (three-channel) instead of RGB
	//!!! DO NOT RELEASE pResults !!!
    TickMeter cvtm;
    cvtm.start();
    cout << "image cols:" << image.cols << " rows:" << image.rows << " step:"<< (int)image.step << endl;
	pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
    cvtm.stop();    
    printf("time = %gms\n", cvtm.getTimeMilli());
    
    printf("%d faces detected.\n", (pResults ? *pResults : 0));
	Mat result_image = image.clone();
	//print the detection results
	for(int i = 0; i < (pResults ? *pResults : 0); i++)
	{
        // short * p = ((short*)(pResults+1))+142*i;
        short * p = ((short*)(pResults+1)) + 12 * size_t(i);
		int confidence = p[0];
		int x = p[1];
		int y = p[2];
		int w = p[3];
		int h = p[4];
        int clsid = int(p[5]);
        
        //show the score of the face. Its range is [0-100]
        char sScore[256];
        snprintf(sScore, 256, "%d", confidence);
        cv::Scalar color = cv::Scalar(0, 255, 0);
        if (clsid == 1)
            color = cv::Scalar(0, 0, 255);
        // cv::putText(result_image, sScore, cv::Point(x, y-3), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        //draw face rectangle
		rectangle(result_image, Rect(x, y, w, h), color, 2);
        
        // print the result
        printf("%s %d: confidence=%d, [%d, %d, %d, %d]\n", labelmap[clsid].c_str(), i, confidence, x, y, w, h);

	}
	imshow("result", result_image);
	waitKey();
    // release the buffer
    free(pBuffer);

	return 0;
}
