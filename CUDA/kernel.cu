#include "opencv2/opencv.hpp"
#include<iostream>
#include<cstdio>
#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
//#include<opencv2/ocl/ocl.hpp>
#include<time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudev/ptr2d/gpumat.hpp"
//#include "opencv2/gpu/gpu.hpp"


using std::cout;
using std::endl;

using namespace cv;


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int height,
	int colorWidthStep,
	int grayWidthStep)
{
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		//Location of gray pixel in output
		const int gray_tid = yIndex * grayWidthStep + xIndex;

		const unsigned char blue = input[color_tid];
		const unsigned char green = input[color_tid + 1];
		const unsigned char red = input[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

int main(int, char**)
{
	cuda::setDevice(0);
	cuda::printCudaDeviceInfo(0);
	int gpucount = cuda::getCudaEnabledDeviceCount();
	if (gpucount != 0) {
		cout << "no. of gpu = " << gpucount << endl;
	}
	else
	{
		cout << "There is no CUDA supported GPU" << endl;
		return -1;
	}


	cv::VideoCapture cap("video.mp4");  // open the default camera
	if (!cap.isOpened()) {  // check if we succeeded
		return -1;
	}

	cv::UMat gray;
	cv::namedWindow("GPU", 1);


	int num_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
	const clock_t begin_time = clock();

	cuda::GpuMat frame_gpu;
	cv::UMat host;
	cv::UMat frame;

	

	for (int i = 0; i<num_frames; i++) 
	{		
		
		cap >> frame; // get a new frame from camera
		//frame_gpu.upload(frame);
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		
		//frame_gpu.download(host);
		//cv::imshow("Display GPU", host);
		//cv::imshow("Display CPU", frame);
		//imshow("GPU", gray);
		if (cv::waitKey(1) >= 0) break;
	}

	printf("GPU:");
	std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC;
	// the camera will be deinitialized automatically in VideoCapture destructor
	

	


	cv::VideoCapture cap2("video.mp4");  // open the default camera
	if (!cap2.isOpened()) {  // check if we succeeded
		return -1;
	}

	cv::Mat gray2;
	cv::namedWindow("CPU", 1);



	int num_frames2 = static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_COUNT));


	const clock_t begin_time2 = clock();

	for (int i = 0; i<num_frames2; i++)
	{
		cv::Mat frame2;
		cap2 >> frame2; // get a new frame from camera

		cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

		//imshow("CPU", gray2);
		if (cv::waitKey(1) >= 0) break;
	}

	printf(" CPU:");
	std::cout << float(clock() - begin_time2) / CLOCKS_PER_SEC;
	// the camera will be deinitialized automatically in VideoCapture destructor

	waitKey(5000);

	return 0;
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	//Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	//Launch the color conversion kernel
	bgr_to_gray_kernel << <grid, block >> >(d_input, d_output, input.cols, input.rows, input.step, output.step);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}