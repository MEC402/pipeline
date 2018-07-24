#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

// Include intel TBB if we're not running on a CUDA system (CUDA is currently Windows only, TBB Linux only)
#ifdef __linux__
#include <unistd.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include "CImg.h"
using namespace cimg_library;

// Input parameters
int iflag, oflag, hflag, rflag, cflag;
char *ivalue, *ovalue;
int edge = 512;

#ifdef _WIN32
inline static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__
void OutImgToXYZ(int i, int j, int face, int edge, double *x, double *y, double *z) {
	auto a = 2.0 * i / edge;
	auto b = 2.0 * j / edge;

	if (face == 0) { // back
		*x = -1;
		*y = 1 - a;
		*z = 3 - b;
	}
	else if (face == 1) { // left
		*x = a - 3;
		*y = -1;
		*z = 3 - b;
	}
	else if (face == 2) { // front
		*x = 1;
		*y = a - 5;
		*z = 3 - b;
	}
	else if (face == 3) { // right
		*x = 7 - a;
		*y = 1;
		*z = 3 - b;
	}
	else if (face == 4) { // top
		*x = b - 1;
		*y = a - 5;
		*z = 1;
	}
	else if (face == 5) { // bottom
		*x = 5 - b;
		*y = a - 5;
		*z = -1;
	}
}

__global__
void ConvertBack(unsigned char *imgIn, unsigned char *front, unsigned char *back,
	unsigned char *left, unsigned char *right, unsigned char *top, unsigned char *bottom,
	int width, int height, int rvalue)
{
	long TstripWidth = rvalue * 4; // Use long in case we're using gigantic 32k+ images
	int edge = rvalue;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = (blockDim.x * gridDim.x);

	int i = index;
	int maxIndex = 0;
	for (; i < TstripWidth; i += stride) {
		int face = int(i / edge);
		int start = edge;
		int end = 2 * edge;

		if (i >= 2 * edge && i < 3 * edge) {
			start = 0;
			end = 3 * edge;
		}

		for (int j = start; j < end; j++) {
			if (j < edge) {
				face = 4;
			}
			else if (j > 2 * edge) {
				face = 5;
			}
			else {
				face = int(i / edge);
			}
			double x, y, z;
			OutImgToXYZ(i, j, face, edge, &x, &y, &z);

			unsigned char *facePtr;
			switch (face) {
			case 0:
				facePtr = back;
				break;
			case 1:
				facePtr = left;
				break;
			case 2:
				facePtr = front;
				break;
			case 3:
				facePtr = right;
				break;
			case 4:
				facePtr = top;
				break;
			case 5:
				facePtr = bottom;
				break;
			}

			double theta = atan2(y, x);
			double r = hypot(x, y);
			double phi = atan2(z, r);
			double uf = (theta + M_PI) / M_PI * height;
			double vf = (M_PI_2 - phi) / M_PI * height;
			int ui = min(static_cast<int>(std::floor(uf)), width);
			int vi = min(static_cast<int>(std::floor(vf)), height);
			int u2 = min(ui + 1, width);
			int v2 = min(vi + 1, height);
			double mu = uf - ui, nu = vf - vi;
			mu = nu = 0;

			// This is the old "read" and "mix" operations unraveled
			// Take first R from ui/vi then mix with second R from u2/vi
			// Repeat for G/B and RGB again for ui/v2 u2/v2
			unsigned char Ra = unsigned char(imgIn[ui + vi * width + 0 * width*height] + (imgIn[u2 + vi * width + 0 * width*height] - imgIn[ui + vi * width + 0 * width*height]) * mu);
			unsigned char Ga = unsigned char(imgIn[ui + vi * width + 1 * width*height] + (imgIn[u2 + vi * width + 1 * width*height] - imgIn[ui + vi * width + 1 * width*height]) * mu);
			unsigned char Ba = unsigned char(imgIn[ui + vi * width + 2 * width*height] + (imgIn[u2 + vi * width + 2 * width*height] - imgIn[ui + vi * width + 2 * width*height]) * mu);
			unsigned char Rb = unsigned char(imgIn[ui + v2 * width + 0 * width*height] + (imgIn[u2 + v2 * width + 0 * width*height] - imgIn[ui + v2 * width + 0 * width*height]) * mu);
			unsigned char Gb = unsigned char(imgIn[ui + v2 * width + 1 * width*height] + (imgIn[u2 + v2 * width + 1 * width*height] - imgIn[ui + v2 * width + 1 * width*height]) * mu);
			unsigned char Bb = unsigned char(imgIn[ui + v2 * width + 2 * width*height] + (imgIn[u2 + v2 * width + 2 * width*height] - imgIn[ui + v2 * width + 2 * width*height]) * mu);
			// Finally mix Ra/Rb etc together for finally interpolated color
			unsigned char R = Ra + (Rb - Ra) * nu;
			unsigned char G = Ga + (Gb - Ga) * nu;
			unsigned char B = Ba + (Bb - Ba) * nu;

			// Based on T-Strip coordinates, mod to edge size and insert into appropriate face
			int idx = ((i%edge) + (j%edge)*edge);
			// CImg uses planar RGBA storage, hence n*edge*edge for each value
			facePtr[idx + 0 * edge*edge] = R;
			facePtr[idx + 1 * edge*edge] = G;
			facePtr[idx + 2 * edge*edge] = B;
			facePtr[idx + 3 * edge*edge] = 255;
		}
	}
}

__device__
void GetFaceStartEnd(int face, int srcWidth, int srcHeight, int edgeSize, 
	int *xStart, int *xEnd, int *yStart, int *yEnd)
{
	// back, left, front, right, top, bottom
	switch (face) {
	case 0: // back
		*xStart = 0;
		*xEnd = edgeSize;
		*yStart = edgeSize;
		*yEnd = edgeSize * 2;
		break;
	case 1: // left
		*xStart = edgeSize;
		*xEnd = edgeSize*2;
		*yStart = edgeSize;
		*yEnd = edgeSize * 2;
		break;
	case 2:
		*xStart = edgeSize * 2;
		*xEnd = edgeSize * 3;
		*yStart = edgeSize;
		*yEnd = edgeSize * 2;
		break;
	case 3:
		*xStart = edgeSize * 3;
		*xEnd = edgeSize * 4;
		*yStart = edgeSize;
		*yEnd = edgeSize * 2;
		break;
	case 4:
		*xStart = edgeSize * 2;
		*xEnd = edgeSize * 3;
		*yStart = 0;
		*yEnd = edgeSize;
		break;
	case 5:
		*xStart = edgeSize * 2;
		*xEnd = edgeSize * 3;
		*yStart = edgeSize * 2;
		*yEnd = edgeSize * 3;
		break;

	}
}

__global__
void ConvertFace(unsigned char *imgIn, unsigned char *imgOut, int face, int width, int height, int rvalue)
{
	long TstripWidth = rvalue * 4; // Use long in case we're using gigantic 32k+ images
	int edge = rvalue;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = (blockDim.x * gridDim.x);
	int xStart, xEnd, yStart, yEnd;
	GetFaceStartEnd(face, width, height, edge, &xStart, &xEnd, &yStart, &yEnd);

	int i = index + xStart;
	int maxIndex = 0;
	for (; i < xEnd; i += stride) {
		for (int j = yStart; j < yEnd; j++) {
			
			double x, y, z;
			OutImgToXYZ(i, j, face, edge, &x, &y, &z);

			double theta = atan2(y, x);
			double r = hypot(x, y);
			double phi = atan2(z, r);
			double uf = (theta + M_PI) / M_PI * height;
			double vf = (M_PI_2 - phi) / M_PI * height;
			int ui = min(static_cast<int>(std::floor(uf)), width);
			int vi = min(static_cast<int>(std::floor(vf)), height);
			int u2 = min(ui + 1, width);
			int v2 = min(vi + 1, height);
			double mu = uf - ui, nu = vf - vi;
			mu = nu = 0;

			// This is the old "read" and "mix" operations unraveled
			// Take first R from ui/vi then mix with second R from u2/vi
			// Repeat for G/B and RGB again for ui/v2 u2/v2
			unsigned char Ra = unsigned char(imgIn[ui + vi * width + 0 * width*height] + (imgIn[u2 + vi * width + 0 * width*height] - imgIn[ui + vi * width + 0 * width*height]) * mu);
			unsigned char Ga = unsigned char(imgIn[ui + vi * width + 1 * width*height] + (imgIn[u2 + vi * width + 1 * width*height] - imgIn[ui + vi * width + 1 * width*height]) * mu);
			unsigned char Ba = unsigned char(imgIn[ui + vi * width + 2 * width*height] + (imgIn[u2 + vi * width + 2 * width*height] - imgIn[ui + vi * width + 2 * width*height]) * mu);
			unsigned char Rb = unsigned char(imgIn[ui + v2 * width + 0 * width*height] + (imgIn[u2 + v2 * width + 0 * width*height] - imgIn[ui + v2 * width + 0 * width*height]) * mu);
			unsigned char Gb = unsigned char(imgIn[ui + v2 * width + 1 * width*height] + (imgIn[u2 + v2 * width + 1 * width*height] - imgIn[ui + v2 * width + 1 * width*height]) * mu);
			unsigned char Bb = unsigned char(imgIn[ui + v2 * width + 2 * width*height] + (imgIn[u2 + v2 * width + 2 * width*height] - imgIn[ui + v2 * width + 2 * width*height]) * mu);
			// Finally mix Ra/Rb etc together for finally interpolated color
			unsigned char R = Ra + (Rb - Ra) * nu;
			unsigned char G = Ga + (Gb - Ga) * nu;
			unsigned char B = Ba + (Bb - Ba) * nu;

			// Based on T-Strip coordinates, mod to edge size and insert into appropriate face
			int idx = ((i%edge) + (j%edge)*edge);
			// CImg uses planar RGBA storage, hence n*edge*edge for each value
			imgOut[idx + 0 * edge*edge] = R;
			imgOut[idx + 1 * edge*edge] = G;
			imgOut[idx + 2 * edge*edge] = B;
			imgOut[idx + 3 * edge*edge] = 255;
		}
	}
}
#endif

void ImgToXYZ(int i, int j, int face, int edge, double *x, double *y, double *z) {
	auto a = 2.0 * i / edge;
	auto b = 2.0 * j / edge;

	if (face == 0) { // back
		*x = -1;
		*y = 1 - a;
		*z = 3 - b;
	}
	else if (face == 1) { // left
		*x = a - 3;
		*y = -1;
		*z = 3 - b;
	}
	else if (face == 2) { // front
		*x = 1;
		*y = a - 5;
		*z = 3 - b;
	}
	else if (face == 3) { // right
		*x = 7 - a;
		*y = 1;
		*z = 3 - b;
	}
	else if (face == 4) { // top
		*x = b - 1;
		*y = a - 5;
		*z = 1;
	}
	else if (face == 5) { // bottom
		*x = 5 - b;
		*y = a - 5;
		*z = -1;
	}
}

/**
**	Convert panorama using an inverse pixel transformation on CPU
**/
void ConvertCPU(unsigned char *imgIn, unsigned char **imgOut, int width, int height) {
	int TotalWidth = edge * 4; // Total width of the T-Strip image
	int start = 0;
	int end = 0;

							   // i/j are T-Strip coordinates, *not* source image coordinates
#ifdef _WIN32
							   // Pardon the sloppy indent on these two loops, it just makes it easier to read with the macros
	for (int i = 0; i < TotalWidth; i++) {
#elif __linux__
	tbb::parallel_for(blocked_range<size_t>(0, TotalWidth, 1), [&](const blocked_range<size_t>& range) {
		for (size_t i = range.begin(); i < range.end(); i++) {
#endif
			int face = int(i / edge);
			start = (i >= 2 * edge && i < 3 * edge) ? 0 : edge;
			end = (i >= 2 * edge && i < 3 * edge) ? edge * 3 : edge * 2;

			// Range start/end determine where in the T-strip to look vertically
			for (int j = start; j < end; ++j) {
				if (j < edge) { // Check if we're above the middle of the strip, then it's the top face
					face = 4;
				}
				else if (j > 2 * edge) { // If we're below the middle of the strip, bottom face
					face = 5;
				}
				else {
					face = int(i / edge); // In the middle of the strip, determine by ratio what face we have
				}

				// Covert T-Strip coordinates to unit-cube coordinates
				double x, y, z;
				ImgToXYZ(i, j, face, edge, &x, &y, &z);

				// Convert unit-cube coordinates into projected unit-sphere coordinates
				double theta = std::atan2(y, x);
				double r = std::hypot(x, y);
				double phi = std::atan2(z, r);
				double uf = (theta + M_PI) / M_PI * height;
				double vf = (M_PI_2 - phi) / M_PI * height;
				int ui = std::min(static_cast<int>(std::floor(uf)), width);
				int vi = std::min(static_cast<int>(std::floor(vf)), height);
				int u2 = std::min(ui + 1, width);
				int v2 = std::min(vi + 1, height);
				double mu = uf - ui, nu = vf - vi;
				mu = nu = 0;

				// This is the old "read" and "mix" operations unraveled
				// Take first R from ui/vi then mix with second R from u2/vi
				// Repeat for G/B and RGB again for ui/v2 u2/v2
				unsigned char Ra = unsigned char(imgIn[ui + vi * width + 0 * width*height] + (imgIn[u2 + vi * width + 0 * width*height] - imgIn[ui + vi * width + 0 * width*height]) * mu);
				unsigned char Ga = unsigned char(imgIn[ui + vi * width + 1 * width*height] + (imgIn[u2 + vi * width + 1 * width*height] - imgIn[ui + vi * width + 1 * width*height]) * mu);
				unsigned char Ba = unsigned char(imgIn[ui + vi * width + 2 * width*height] + (imgIn[u2 + vi * width + 2 * width*height] - imgIn[ui + vi * width + 2 * width*height]) * mu);
				unsigned char Rb = unsigned char(imgIn[ui + v2 * width + 0 * width*height] + (imgIn[u2 + v2 * width + 0 * width*height] - imgIn[ui + v2 * width + 0 * width*height]) * mu);
				unsigned char Gb = unsigned char(imgIn[ui + v2 * width + 1 * width*height] + (imgIn[u2 + v2 * width + 1 * width*height] - imgIn[ui + v2 * width + 1 * width*height]) * mu);
				unsigned char Bb = unsigned char(imgIn[ui + v2 * width + 2 * width*height] + (imgIn[u2 + v2 * width + 2 * width*height] - imgIn[ui + v2 * width + 2 * width*height]) * mu);
				// Finally mix Ra/Rb etc together for finally interpolated color
				unsigned char R = Ra + (Rb - Ra) * nu;
				unsigned char G = Ga + (Gb - Ga) * nu;
				unsigned char B = Ba + (Bb - Ba) * nu;

				// Based on T-Strip coordinates, mod to edge size and insert into appropriate face
				int idx = ((i%edge) + (j%edge)*edge);
				// CImg uses planar RGBA storage, hence n*edge*edge for each value
				imgOut[face][idx + 0 * edge*edge] = R;
				imgOut[face][idx + 1 * edge*edge] = G;
				imgOut[face][idx + 2 * edge*edge] = B;
				imgOut[face][idx + 3 * edge*edge] = 255;
			}
		}
#ifdef __linux__
	});
#endif
}


int parseParameters(int argc, char *argv[]) {
	iflag = oflag = hflag = rflag = cflag = 0;
	ivalue = ovalue = NULL;
	int c;

#ifdef _WIN32
	for (int i = 1; i < argc; i++) {
		if (argv[i] == std::string("-i")) {
			iflag = 1;
			ivalue = argv[++i];
		}
		if (argv[i] == std::string("-o")) {
			oflag = 1;
			ovalue = argv[++i];
		}
		if (argv[i] == std::string("-r")) {
			rflag = 1;
			edge = std::stoi(argv[++i]);
		}
		if (argv[i] == std::string("-c")) {
			cflag = 1;
		}
		if (argv[i] == std::string("-h")) {
			fprintf(stderr, "Usage:\n\t -i <input file>\n\t -o <output file(s)>\n\t -r <edge size>\n\t -c (enable CUDA)\n");
			abort();
		}
	}
#elif __linux__
	opterr = 0;
	while ((c = getopt(argc, argv, "i:o:r:")) != -1)
		switch (c) {
		case 'i':
			// input file
			iflag = 1;
			ivalue = optarg;
			break;
		case 'o':
			oflag = 1;
			ovalue = optarg;
			break;
		case 'r':
			rflag = 1;
			edge = std::stoi(optarg);
			break;
		case '?':
			if (optopt == 'i' || optopt == 'o' || optopt == 'r')
				fprintf(stderr, "Option -%c requires an argument.\n", optopt);
			else if (isprint(optopt))
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
			return 1;
		default:
			abort();
		}
#endif

	if (iflag == 0 || oflag == 0) {
		fprintf(stderr, "No inputs or outputs specified: %d / %d\n", iflag, oflag);
		abort();
		return 1;
	}
	return 0;
}


int main(int argc, char *argv[])
{

	parseParameters(argc, argv);

	CImg<unsigned char> ImgIn(ivalue);

	// Create output images
	CImg<unsigned char>* CImgOut[6];
	unsigned char* imgOut[6];
	for (int i = 0; i < 6; ++i) {
		CImgOut[i] = new CImg<unsigned char>(edge, edge, 1, 4, 255);
		imgOut[i] = (unsigned char*)CImgOut[i]->data();
	}
	std::chrono::high_resolution_clock::time_point total = std::chrono::high_resolution_clock::now();
	if (cflag) {
		int InSize = ImgIn.size();
		int width = ImgIn.width();
		int height = ImgIn.height();
		unsigned char *d_ImgIn, *d_ImgFront, *d_ImgBack, *d_ImgLeft, *d_ImgRight, *d_ImgTop, *d_ImgBottom;
		HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgIn, InSize * sizeof(unsigned char)));
		std::memcpy(d_ImgIn, ImgIn.data(), InSize * sizeof(unsigned char));
		ImgIn.clear();

		size_t total_memory, free_memory;
		HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory) );
		printf("Total Memory: %lld\n Free Memory: %lld\n", total_memory, free_memory);

		long outSize = edge * edge * 4;
		printf("Insize memory: %d\n Outsize memory each: %lld\n Outsize memory total: %lld\n", InSize, outSize, outSize * 6);

		if (outSize > free_memory) {
			fprintf(stderr, "Not enough memory free on GPU device, please run without -c flag to compute on CPU");
			return -1;
		}

		if (outSize * 6 > free_memory) {
			//std::thread threads[6];
			fprintf(stderr, "Not enough memory on device to run all faces in parallel, running one face at a time\n");
			unsigned char *d_ImgOut, *h_ImgOut;
			//HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgOut, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMalloc((void**)&d_ImgOut, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMallocHost((void**)&h_ImgOut, outSize * sizeof(unsigned char)));
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			int blockSize = 256;
			int blockCount = (edge + blockSize - 1) / blockSize;
			printf("Total threads: %d\n", blockSize * blockCount);
			for (int i = 0; i < 6; i++) {
				fprintf(stderr, "Starting face %d...\n", i);
				ConvertFace <<<blockCount, blockSize >>> (d_ImgIn, d_ImgOut, i, width, height, edge);
				HANDLE_ERROR(cudaDeviceSynchronize());
				HANDLE_ERROR(cudaThreadSynchronize());
				//std::memcpy(imgOut[i], d_ImgOut, outSize * sizeof(unsigned char));
				HANDLE_ERROR(cudaMemcpy(h_ImgOut, d_ImgOut, outSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
				std::memcpy(imgOut[i], h_ImgOut, outSize * sizeof(unsigned char));
				std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
				CImgOut[i]->save_png(fname.c_str());
				CImgOut[i]->clear();
				//threads[i] = std::thread([&]() {
				//	std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
				//	CImgOut[i]->save_png(fname.c_str());
				//	CImgOut[i]->clear();
				//});
			}
			HANDLE_ERROR(cudaFree(d_ImgIn));
			HANDLE_ERROR(cudaFree(d_ImgOut));
			HANDLE_ERROR(cudaFreeHost(h_ImgOut));
			fprintf(stderr, "Time to convert: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());
			//for (int i = 0; i < 6; i++) {
			//	threads[i].join();
			//}
			//for (int i = 0; i < 6; i++) {
			//	std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
			//	CImgOut[i]->save_png(fname.c_str());
			//}
			fprintf(stderr, "Total Time To Convert And Write: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total).count());
		}
		else {

			HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgFront, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
			printf("Free Memory: %lld\n", free_memory);

			HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgBack, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
			printf("Free Memory: %lld\n", free_memory);

			HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgLeft, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
			printf("Free Memory: %lld\n", free_memory);

			HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgRight, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
			printf("Free Memory: %lld\n", free_memory);

			HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgTop, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
			printf("Free Memory: %lld\n", free_memory);

			HANDLE_ERROR(cudaMallocManaged((void**)&d_ImgBottom, outSize * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
			printf("Free Memory: %lld\n", free_memory);

			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			int blockSize = 256;
			int blockCount = (ImgIn.width() + blockSize - 1) / blockSize;
			fprintf(stderr, "Starting...\n");
			ConvertBack << <blockCount, blockSize >> > (d_ImgIn, d_ImgFront, d_ImgBack, d_ImgLeft, d_ImgRight, d_ImgTop, d_ImgBottom, ImgIn.width(), ImgIn.height(), edge);
			HANDLE_ERROR(cudaDeviceSynchronize());
			HANDLE_ERROR(cudaFree(d_ImgIn));
			fprintf(stderr, "Time to convert: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());
			fprintf(stderr, "Synchronized\n");


			// TODO: Loop this
			std::string filename = std::string(ovalue); // This is so stupid
			std::string copy = filename;
			copy.append("_0.png");
			printf("Writing Back...\n");
			std::memcpy(imgOut[0], d_ImgBack, outSize * sizeof(unsigned char));
			HANDLE_ERROR(cudaFree(d_ImgBack));
			CImgOut[0]->save_png(copy.c_str());

			printf("Writing Front...\n");
			copy = filename;
			copy.append("_1.png");
			std::memcpy(imgOut[1], d_ImgFront, outSize * sizeof(unsigned char));
			HANDLE_ERROR(cudaFree(d_ImgFront));
			CImgOut[1]->save_png(copy.c_str());

			printf("Writing Left...\n");
			copy = filename;
			copy.append("_2.png");
			std::memcpy(imgOut[2], d_ImgLeft, outSize * sizeof(unsigned char));
			HANDLE_ERROR(cudaFree(d_ImgLeft));
			CImgOut[2]->save_png(copy.c_str());

			printf("Writing Right...\n");
			copy = filename;
			copy.append("_3.png");
			std::memcpy(imgOut[3], d_ImgRight, outSize * sizeof(unsigned char));
			HANDLE_ERROR(cudaFree(d_ImgRight));
			CImgOut[3]->save_png(copy.c_str());

			printf("Writing Top...\n");
			copy = filename;
			copy.append("_4.png");
			std::memcpy(imgOut[4], d_ImgTop, outSize * sizeof(unsigned char));
			HANDLE_ERROR(cudaFree(d_ImgTop));
			CImgOut[4]->save_png(copy.c_str());

			printf("Writing Bottom...\n");
			copy = filename;
			copy.append("_5.png");
			std::memcpy(imgOut[5], d_ImgBottom, outSize * sizeof(unsigned char));
			HANDLE_ERROR(cudaFree(d_ImgBottom));
			CImgOut[5]->save_png(copy.c_str());
			fprintf(stderr, "Total Time To Convert And Write: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total).count());
		}
	}
	else {
		ConvertCPU(ImgIn.data(), imgOut, ImgIn.width(), ImgIn.height());

		for (int i = 0; i < 6; i++) {
			std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
			CImgOut[i]->save_png(fname.c_str());
		}
		fprintf(stderr, "Total Time To Convert And Write: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total).count());
	}

	return 0;
}



