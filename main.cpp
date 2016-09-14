/*******************************************************************
*   main.cpp
*   CLATCH
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Sep 12, 2016
*******************************************************************/
//
// Fastest implementation of the fully scale-
// and rotation-invariant LATCH 512-bit binary
// feature descriptor as described in the 2015
// paper by Levi and Hassner:
//
// "LATCH: Learned Arrangements of Three Patch Codes"
// http://arxiv.org/abs/1501.03719
//
// See also the ECCV 2016 Descriptor Workshop paper, of which I am a coauthor:
//
// "The CUDA LATCH Binary Descriptor"
// http://arxiv.org/abs/1609.03986
//
// And the original LATCH project's website:
// http://www.openu.ac.il/home/hassner/projects/LATCH/
//
// This implementation is insanely fast, matching or beating
// the much simpler ORB descriptor despite outputting twice
// as many bits AND being a superior descriptor.
//
// A key insight responsible for much of the performance of
// this laboriously crafted CUDA kernel is due to
// Christopher Parker (https://github.com/csp256), to whom
// I am extremely grateful.
//
// CUDA CC 3.0 or higher is required.
//
// All functionality is contained in the files CLATCH.h
// and CLATCH.cu. 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#include <chrono>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "CLATCH.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

int main() {
	// ------------- Configuration ------------
	constexpr int warmups = 100;
	constexpr int runs = 500;
	constexpr int numkps = 5000;
	constexpr char name[] = "test.jpg";
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	// --------------------------------


	// ------------- Detection ------------
	std::cout << std::endl << "Detecting..." << std::endl;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(numkps, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
	std::vector<cv::KeyPoint> keypoints;
	orb->detect(image, keypoints);
	keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [image](const cv::KeyPoint& kp) {return kp.pt.x <= 36 || kp.pt.y <= 36 || kp.pt.x >= image.cols - 36 || kp.pt.y >= image.rows - 36; }), keypoints.end());
	// --------------------------------


	// ------------- CLATCH ------------

	// arranging keypoints for PCI transfer
	std::vector<KeyPoint> kps;
	for (const auto& kp : keypoints) kps.emplace_back(kp.pt.x, kp.pt.y, kp.size, kp.angle * 3.14159265f / 180.0f);
	
	// setting cache and shared modes
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

	// allocating space for descriptors
	uint64_t* d_desc;
	cudaMalloc(&d_desc, 64 * kps.size());

	// allocating and transferring keypoints and binding to texture object
	KeyPoint* d_kps;
	cudaMalloc(&d_kps, kps.size() * sizeof(KeyPoint));
	cudaMemcpy(d_kps, &kps[0], kps.size() * sizeof(KeyPoint), cudaMemcpyHostToDevice);

	// allocating and transferring triplets and binding to texture object
	uint32_t* d_triplets;
	cudaMalloc(&d_triplets, 2048 * sizeof(uint16_t));
	cudaMemcpy(d_triplets, triplets, 2048 * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaChannelFormatDesc chandesc_trip = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
	cudaArray* d_trip_arr;
	cudaMallocArray(&d_trip_arr, &chandesc_trip, 512);
	cudaMemcpyToArray(d_trip_arr, 0, 0, d_triplets, 2048 * sizeof(uint16_t), cudaMemcpyHostToDevice);
	struct cudaResourceDesc resdesc_trip;
	memset(&resdesc_trip, 0, sizeof(resdesc_trip));
	resdesc_trip.resType = cudaResourceTypeArray;
	resdesc_trip.res.array.array = d_trip_arr;
	struct cudaTextureDesc texdesc_trip;
	memset(&texdesc_trip, 0, sizeof(texdesc_trip));
	texdesc_trip.addressMode[0] = cudaAddressModeClamp;
	texdesc_trip.filterMode = cudaFilterModePoint;
	texdesc_trip.readMode = cudaReadModeElementType;
	texdesc_trip.normalizedCoords = 0;
	cudaTextureObject_t d_trip_tex = 0;
	cudaCreateTextureObject(&d_trip_tex, &resdesc_trip, &texdesc_trip, nullptr);

	// allocating and transferring image and binding to texture object
	cudaChannelFormatDesc chandesc_img = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray* d_img_arr;
	cudaMallocArray(&d_img_arr, &chandesc_img, image.cols, image.rows);
	cudaMemcpyToArray(d_img_arr, 0, 0, image.data, image.rows * image.cols, cudaMemcpyHostToDevice);
	struct cudaResourceDesc resdesc_img;
	memset(&resdesc_img, 0, sizeof(resdesc_img));
	resdesc_img.resType = cudaResourceTypeArray;
	resdesc_img.res.array.array = d_img_arr;
	struct cudaTextureDesc texdesc_img;
	memset(&texdesc_img, 0, sizeof(texdesc_img));
	texdesc_img.addressMode[0] = cudaAddressModeClamp;
	texdesc_img.addressMode[1] = cudaAddressModeClamp;
	texdesc_img.filterMode = cudaFilterModePoint;
	texdesc_img.readMode = cudaReadModeElementType;
	texdesc_img.normalizedCoords = 0;
	cudaTextureObject_t d_img_tex = 0;
	cudaCreateTextureObject(&d_img_tex, &resdesc_img, &texdesc_img, nullptr);

	std::cout << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) CLATCH(d_img_tex, d_trip_tex, d_kps, static_cast<int>(kps.size()), d_desc);
	std::cout << "Testing..." << std::endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CLATCH(d_img_tex, d_trip_tex, d_kps, static_cast<int>(kps.size()), d_desc);
	high_resolution_clock::time_point end = high_resolution_clock::now();
	// --------------------------------

	std::cout << std::endl << "CLATCH took " << static_cast<double>((end - start).count()) * 1e-3 / (static_cast<double>(runs) * static_cast<double>(kps.size())) << " us per desc over " << kps.size() << " desc" << (kps.size() == 1 ? "." : "s.") << std::endl << std::endl;
	
	uint64_t* h_GPUdesc = new uint64_t[8 * kps.size()];
	cudaMemcpy(h_GPUdesc, d_desc, 64 * kps.size(), cudaMemcpyDeviceToHost);

	std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	long long total = 0;
	for (size_t i = 0; i < 8 * kps.size(); ++i) total += h_GPUdesc[i];
	std::cout << "Checksum: " << std::hex << total << std::endl << std::endl;
}
