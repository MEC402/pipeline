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
#include "png.h"
#endif

#ifdef __linux__
#include <unistd.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

// If on windows: Make sure libpng nuget package is installed to make use of native png saving
// If on linux: Make sure libpng is linked against when compiling
#define cimg_use_png
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

// Overload for nesting Interpolate calls
__device__
inline unsigned char LinearInterpolate(float weight, unsigned char v1, unsigned char v2)
{
	return unsigned char(weight*v2 + (1.0f - weight) * v1);
}

// Expectes 1 weight and 2 values
__device__
inline unsigned char LinearInterpolate(float weight, unsigned char *values)
{
	return unsigned char(weight * (values[1]) + (1.0f - weight)*values[0]);
}

// Expects 2 weights and 4 values
__device__
inline unsigned char BilinearInterpolate(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		LinearInterpolate(weight[1], &values[0]),
		LinearInterpolate(weight[1], &values[2])
	};
	return LinearInterpolate(weight[0], prime);
}

// Expects 3 weights and 8 values
__device__
inline unsigned char TrilinearInterpolate(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		BilinearInterpolate(&(weight[0]), &(values[0])),
		BilinearInterpolate(&(weight[1]), &(values[4]))
	};
	return LinearInterpolate(weight[3], prime);
}

// Convert the entire cubemap at once
__global__
void ConvertBack(unsigned char *imgIn, unsigned char **imgOut, int width, int height, int rvalue)
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

			double theta = atan2(y, x);
			double r = hypot(x, y);
			double phi = atan2(z, r);
			double uf = (theta + M_PI) / M_PI * height;
			double vf = (M_PI_2 - phi) / M_PI * height;
			int ui = min(static_cast<int>(std::floor(uf)), width);
			int vi = min(static_cast<int>(std::floor(vf)), height);
			int u2 = min(ui + 1, width);
			int v2 = min(vi + 1, height);
			int u3 = min(ui + 2, width);
			int v3 = min(vi + 2, height);
			int u4 = max(ui - 1, 0);
			int v4 = max(vi - 1, 0);
			int u[4] = { ui, u2, u3, u4 };
			int v[4] = { vi, v2, v3, v4 };

			unsigned char Rval[16];
			unsigned char Gval[16];
			unsigned char Bval[16];

			for (int a = 0; a < 4; a++) {
				for (int b = 0; b < 4; b++) {
					if (u[a] + v[b] * width + 2 * width*height > width*height * 3) {
						printf("%d %d %d %d %d %d\n", a, b, u[a], v[b], width, height, u[a] + v[b] * width + 2 * width*height, width*height * 3);
					}
					Rval[a * 4 + b] = imgIn[u[a] + v[b] * width + 0 * width*height];
					Gval[a * 4 + b] = imgIn[u[a] + v[b] * width + 1 * width*height];
					Bval[a * 4 + b] = imgIn[u[a] + v[b] * width + 2 * width*height];
				}
			}

			float weight[3] = { 0.5f, 0.5f, 0.5f };
			unsigned char R = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Rval[0]), TrilinearInterpolate(weight, &Rval[8]));
			unsigned char G = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Gval[0]), TrilinearInterpolate(weight, &Gval[8]));
			unsigned char B = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Bval[0]), TrilinearInterpolate(weight, &Bval[8]));

			// Based on T-Strip coordinates, mod to edge size and insert into appropriate face
			int idx = ((i%edge) + (j%edge)*edge);
			unsigned char *ptr = imgOut[face];
			// CImg uses planar RGBA storage, hence n*edge*edge for each value
			ptr[idx + 0 * edge*edge] = R;
			ptr[idx + 1 * edge*edge] = G;
			ptr[idx + 2 * edge*edge] = B;
			ptr[idx + 3 * edge*edge] = 255;
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


// Convert faces one at a time if we can't fit everything in memory at once
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
			int u3 = min(ui + 2, width);
			int v3 = min(vi + 2, height);
			int u4 = max(ui - 1, 0);
			int v4 = max(vi - 1, 0);
			int u[4] = { ui, u2, u3, u4 };
			int v[4] = { vi, v2, v3, v4 };
			double mu = uf - ui, nu = vf - vi;
			mu = nu = 0;

			unsigned char Rval[16];
			unsigned char Gval[16];
			unsigned char Bval[16];

			for (int a = 0; a < 4; a++) {
				for (int b = 0; b < 4; b++) {
					Rval[a * 4 + b] = imgIn[u[a] + v[b] * width + 0 * width*height];
					Gval[a * 4 + b] = imgIn[u[a] + v[b] * width + 1 * width*height];
					Bval[a * 4 + b] = imgIn[u[a] + v[b] * width + 2 * width*height];
				}
			}
			float weight[3] = { 0.5f, 0.5f, 0.5f };
			unsigned char R = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Rval[0]), TrilinearInterpolate(weight, &Rval[8]));
			unsigned char G = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Gval[0]), TrilinearInterpolate(weight, &Gval[8]));
			unsigned char B = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Bval[0]), TrilinearInterpolate(weight, &Bval[8]));

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



// Overload for nesting Interpolate calls
unsigned char Linear(float weight, unsigned char v1, unsigned char v2)
{
	return unsigned char(weight*v2 + (1.0f - weight) * v1);
}

// Expectes 1 weight and 2 values
unsigned char Linear(float weight, unsigned char *values)
{
	return unsigned char(weight * (values[1]) + (1.0f - weight)*values[0]);
}

// Expects 2 weights and 4 values
unsigned char Bilinear(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		Linear(weight[1], &values[0]),
		Linear(weight[1], &values[2])
	};
	return Linear(weight[0], prime);
}

// Expects 3 weights and 8 values
unsigned char Trilinear(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		Bilinear(&(weight[0]), &(values[0])),
		Bilinear(&(weight[1]), &(values[4]))
	};
	return Linear(weight[3], prime);
}


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

// Convert panorama using an inverse pixel transformation on CPU
void ConvertCPU(unsigned char *imgIn, unsigned char **imgOut, int width, int height) {
	int TotalWidth = edge * 4; // Total width of the T-Strip image
	int start = 0;
	int end = 0;

	// i/j are T-Strip coordinates, that then get converted to XYZ spherical projection coordinates
#ifdef __linux__
	tbb::parallel_for(tbb::blocked_range<size_t>(0, TotalWidth, 1), [&](const tbb::blocked_range<size_t>& range) {
		for (size_t i = range.begin(); i < range.end(); i++) {
#endif
#ifdef _WIN32
	for (int i = 0; i < TotalWidth; i++) {
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

				double theta = std::atan2(y, x);
				double r = std::hypot(x, y);
				double phi = std::atan2(z, r);
				double uf = (theta + M_PI) / M_PI * height;
				double vf = (M_PI_2 - phi) / M_PI * height;
				/*
					Coordinate structure:
					[     ][     ][     ][u3/v4]
					[     ][     ][u2/v2][     ]
					[     ][ui/vi][     ][     ]
					[u4/v4][     ][     ][     ]
				*/
				int ui = std::min(static_cast<int>(std::floor(uf)), width);
				int vi = std::min(static_cast<int>(std::floor(vf)), height);
				int u2 = std::min(ui + 1, width);
				int v2 = std::min(vi + 1, height);
				int u3 = std::min(ui + 2, width);
				int v3 = std::min(vi + 2, height-1); // Some weird bug here, if v3 == width we step outside memory bounds.  It shouldn't matter because we're sampling 16 coords hopefully
				int u4 = std::max(ui - 1, 0);
				int v4 = std::max(vi - 1, 0);
				int u[4] = { ui, u2, u3, u4 };
				int v[4] = { vi, v2, v3, v4 };

				unsigned char Rval[16];
				unsigned char Gval[16];
				unsigned char Bval[16];

				for (int a = 0; a < 4; a++) {
					for (int b = 0; b < 4; b++) {
						Rval[a * 4 + b] = imgIn[u[a] + v[b] * width + 0 * width*height];
						Gval[a * 4 + b] = imgIn[u[a] + v[b] * width + 1 * width*height];
						Bval[a * 4 + b] = imgIn[u[a] + v[b] * width + 2 * width*height];
					}
				}

				float weight[3] = { 0.5f, 0.5f, 0.5f };
				unsigned char R = Linear(weight[0], Trilinear(weight, &Rval[0]), Trilinear(weight, &Rval[8]));
				unsigned char G = Linear(weight[0], Trilinear(weight, &Gval[0]), Trilinear(weight, &Gval[8]));
				unsigned char B = Linear(weight[0], Trilinear(weight, &Bval[0]), Trilinear(weight, &Bval[8]));

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
		case 'c':
			cflag = 1;
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
	printf("Converting [%s] to faces [%s] with size [%d]...\n", ivalue, ovalue, edge);
	CImg<unsigned char> ImgIn(ivalue);
	printf("%d\n", ImgIn.size());
	// Create output images
	CImg<unsigned char>* CImgOut[6];
	unsigned char* imgOut[6];
	for (int i = 0; i < 6; ++i) {
		CImgOut[i] = new CImg<unsigned char>(edge, edge, 1, 4, 255);
		imgOut[i] = (unsigned char*)CImgOut[i]->data();
	}
	std::chrono::high_resolution_clock::time_point total = std::chrono::high_resolution_clock::now();
	if (cflag) {
		printf("Using CUDA for processing\n");
		int InSize = ImgIn.size();
		int width = ImgIn.width();
		int height = ImgIn.height();
		unsigned char *d_ImgIn;
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
			std::thread threads[6];
			fprintf(stderr, "Not enough memory on device to run all faces in parallel, running one face at a time\n");
			unsigned char *d_ImgOut, *h_ImgOut;

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
				HANDLE_ERROR(cudaMemcpy(h_ImgOut, d_ImgOut, outSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
				std::memcpy(imgOut[i], h_ImgOut, outSize * sizeof(unsigned char));
				threads[i] = std::thread([&, i]() {
					std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
					CImgOut[i]->save_png(fname.c_str());
					CImgOut[i]->clear();
					printf("Thread %d finished writing to disk\n", i);
				});
			}
			HANDLE_ERROR(cudaFree(d_ImgIn));
			HANDLE_ERROR(cudaFree(d_ImgOut));
			HANDLE_ERROR(cudaFreeHost(h_ImgOut));
			fprintf(stderr, "Time to convert: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());
			for (int i = 0; i < 6; i++) {
				threads[i].join();
				printf("Joined thread %d\n", i);
			}
			fprintf(stderr, "Total Time To Convert And Write: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total).count());
		}
		else {
			printf("Allocating GPU memory\n");
			unsigned char **d_Faces;
			HANDLE_ERROR(cudaMallocManaged((void**)&d_Faces, sizeof(unsigned char*) * 6));
			for (int i = 0; i < 6; i++) {
				HANDLE_ERROR(cudaMallocManaged((void**)&d_Faces[i], outSize * sizeof(unsigned char)));
				HANDLE_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
				printf("Free Device Memory: %lldMB\n", free_memory / 1024 / 1024);
			}

			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			int blockSize = 256;
			int blockCount = (width + blockSize - 1) / blockSize;
			fprintf(stderr, "Starting conversion...\n");
			ConvertBack <<<blockCount, blockSize >>> (d_ImgIn, d_Faces, width, height, edge);
			HANDLE_ERROR(cudaDeviceSynchronize());
			HANDLE_ERROR(cudaThreadSynchronize());
			HANDLE_ERROR(cudaFree(d_ImgIn));
			fprintf(stderr, "Time to convert: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());
			fprintf(stderr, "Device Synchronized\n");


			std::thread threads[6];
			for (int i = 0; i < 6; i++) {
				std::memcpy(imgOut[i], d_Faces[i], outSize * sizeof(unsigned char));
				HANDLE_ERROR(cudaFree(d_Faces[i]));
				threads[i] = std::thread([&, i]() {
					std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
					printf("Writing %s to disk\n", fname.c_str());
					CImgOut[i]->save_png(fname.c_str());
					CImgOut[i]->clear();
					printf("Finished writing %s\n", fname.c_str());
				});
			}
			HANDLE_ERROR(cudaFree(d_Faces));
			for (int i = 0; i < 6; i++) {
				threads[i].join();
			}

			fprintf(stderr, "Total Time To Convert And Write: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total).count());
		}
	}
	else {
		printf("Using CPU for Processing\n");
		ConvertCPU(ImgIn.data(), imgOut, ImgIn.width(), ImgIn.height());
		std::thread threads[6];
		for (int i = 0; i < 6; i++) {
			threads[i] = std::thread([&, i]() {
				std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
				printf("Writing %s to disk\n", fname.c_str());
				CImgOut[i]->save_png(fname.c_str());
				CImgOut[i]->clear();
				printf("Finished writing %s\n", fname.c_str());
			});
		}
		for (int i = 0; i < 6; i++) {
			threads[i].join();
		}
		fprintf(stderr, "Total Time To Convert And Write: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total).count());
	}

	return 0;
}



