#ifndef __IMAGEHANDLER_H
#define __IMAGEHANDLER_H

#define GLM_ENABLE_EXPERIMENTAL
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include "tiff.h"
#include <tiffio.h>

#include "png.h"
#include <zlib.h> 

#include <locale>
#include <thread>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "Quad.h"
#include "Filters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


IMGenum SetImagetype(std::string filepath);
Texture ReadImage(IMGenum type, const char *filepath);
void SaveImage(Quad img);

Texture LoadTIFF(const char *filepath);
void SaveTIFF(Quad img);
Texture LoadSTBI(const char *filepath);
void SavePNG(Quad imgA);

void TransformData(Quad img, unsigned char **imgOut);

IMGenum SetImagetype(std::string filepath)
{
	std::locale loc;
	std::string extension = filepath.substr(filepath.find_last_of('.'));
	// STL C++ doesn't have a "string.tolower()" function.
	// You have go to be kidding me.
	for (int i = 0; i < extension.length(); i++)
		extension[i] = std::tolower(extension[i], loc);

	if (!extension.compare(".png"))
		return PNG;
	if (!extension.compare(".tif") || !extension.compare(".tiff"))
		return TIF;
	if (!extension.compare(".jpg") || !extension.compare(".jpeg"))
		return JPG;
}

Texture ReadImage(IMGenum type, const char *filepath)
{
	switch (type) {
	case PNG:
	case JPG:
		return LoadSTBI(filepath);
		break;
	case TIF:
		return LoadTIFF(filepath);
		break;
	}
}

void SaveImage(Quad img)
{
	switch (img.type) {
	case PNG:
		SavePNG(img);
		break;
	case JPG:
		printf("Not implemented\n");
		break;
	case TIF:
		SaveTIFF(img);
		break;
	}
}

Texture LoadTIFF(const char *filepath)
{
	unsigned int width, height, pixels, *raster;
	TIFF *file = TIFFOpen(filepath, "r");
	TIFFGetField(file, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(file, TIFFTAG_IMAGELENGTH, &height);
	pixels = width * height;
	raster = (unsigned int*)_TIFFmalloc(pixels * sizeof(unsigned int));
	int result = TIFFReadRGBAImage(file, width, height, raster, 0);
	if (result == 0)
		printf("Some kind of error occured\n");

	Texture t;
	t.width = width;
	t.height = height;
	t.channels = 3;
	t.data = new unsigned char[width * height * 3];
	unsigned char *ptr = t.data;
	for (int i = 0; i < pixels; i++) {
		*(ptr++) = (unsigned char)TIFFGetR(raster[i]);
		*(ptr++) = (unsigned char)TIFFGetG(raster[i]);
		*(ptr++) = (unsigned char)TIFFGetB(raster[i]);
	}
	_TIFFfree(raster);
	TIFFClose(file);

	return t;
}

void SaveTIFF(Quad img)
{
	printf("Saving TIFF file...\n");
	TIFF *file = TIFFOpen("Out.tif", "w");
	TIFFSetField(file, TIFFTAG_IMAGEWIDTH, img.texture.width);
	TIFFSetField(file, TIFFTAG_IMAGELENGTH, img.texture.height);
	TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, img.texture.channels);
	TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, 8);
	TIFFSetField(file, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

	size_t bytesPerRow = img.texture.channels * img.texture.width;
	unsigned char *buffer = NULL;

	if (TIFFScanlineSize(file) < bytesPerRow)
		buffer = (unsigned char*)_TIFFmalloc(bytesPerRow);
	else
		buffer = (unsigned char*)_TIFFmalloc(TIFFScanlineSize(file));

	TIFFSetField(file, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(file, img.texture.width * img.texture.channels));
	
	unsigned char **imgOut = new unsigned char*[img.texture.height];
	for (int i = 0; i < img.texture.height; i++)
		imgOut[i] = new unsigned char[img.texture.channels*img.texture.width];

	TransformData(img, imgOut);

	for (unsigned int row = 0; row < img.texture.height; row++) {
		std::memcpy(buffer, imgOut[row], bytesPerRow);
		if (TIFFWriteScanline(file, buffer, row, 0) < 0)
			break;
	}
	TIFFClose(file);
	if (buffer)
		_TIFFfree(buffer);

	for (int i = 0; i < img.texture.height; i++)
		delete[]imgOut[i];
	delete[]imgOut;


	printf("TIFF file saved to disk successfully\n");
}

Texture LoadSTBI(const char *filepath)
{
	stbi_set_flip_vertically_on_load(true);
	Texture t;
	t.data = stbi_load(filepath, &t.width, &t.height, &t.channels, 0);
	return t;
}

void SavePNG(Quad imgA)
{
	/* stb_image_write sucks, so write our own simple PNG save routine */
	printf("Gathering image data...\n");

	int width = imgA.texture.width;
	int height = imgA.texture.height;
	int channels = imgA.texture.channels;

	std::FILE *nfile = fopen("Out.png", "wb");
	png_voidp usr_err_ptr = NULL;
	png_error_ptr usr_err_fn = NULL, usr_warn_fn = NULL;
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, usr_err_ptr, usr_err_fn, usr_warn_fn);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, nfile);

	const int bit_depth = 8;
	const int byte_depth = bit_depth >> 3;
	const int pixel_bit_depth_flag = channels * (bit_depth - 1);

	png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(png_ptr, info_ptr);

	png_bytep *const imgOut = new png_byte*[height];
	for (int i = 0; i < height; i++)
		imgOut[i] = new png_byte[byte_depth*channels*width];

	TransformData(imgA, (unsigned char**)imgOut);

	printf("Writing to disk...\n");

	png_write_image(png_ptr, imgOut);
	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);

	fclose(nfile);

	for (int i = 0; i < height; i++)
		delete[]imgOut[i];
	delete[]imgOut;

	printf("Finished writing Out.png\n");
}

void TransformData(Quad img, unsigned char **imgOut)
{
	int width = img.texture.width;
	int height = img.texture.height;
	int channels = img.texture.channels;

	// TODO: Make these work
	glm::mat2 xy_to_lla(
		glm::vec2(2.0f * M_PI / width, 0.0f),
		glm::vec2(0.0f, M_PI / height)
	);

	//glm::mat3 pmat(
	//	glm::vec3(1.0f, 0.0f, 0.0f),
	//	glm::vec3(0.0f,  cos(-img.pitch), sin(-img.pitch)),
	//	glm::vec3(0.0f, -sin(-img.pitch), cos(-img.pitch))
	//);
	//glm::mat3 ymat(
	//	glm::vec3(cos(img.yaw), 0.0f, -sin(img.yaw)),
	//	glm::vec3(0.0f, 1.0f, 0.0f),
	//	glm::vec3(sin(img.yaw), 0.0f, cos(img.yaw))
	//);
	//glm::mat3 rmat(
	//	glm::vec3(cos(img.roll), -sin(img.roll), 0.0f),
	//	glm::vec3(sin(img.roll), cos(img.roll), 0.0f),
	//	glm::vec3(0.0f, 0.0f, 1.0f)
	//);

	// This is just the resulting pmat * ymat * rmat matrix
	// Pitch is inverted to match what we see in the OpenGL render
	glm::mat3 rotate(
		glm::vec3(cos(img.roll)*cos(img.yaw), -sin(img.roll), cos(img.roll)*sin(img.yaw)),
		glm::vec3(sin(img.roll)*cos(img.yaw)*cos(-img.pitch) - sin(img.yaw)*sin(-img.pitch), cos(img.roll)*cos(-img.pitch), sin(img.roll)*sin(img.yaw)*cos(-img.pitch)+cos(img.yaw)*sin(-img.pitch)),
		glm::vec3(-sin(img.roll)*cos(img.yaw)*sin(-img.pitch)-sin(img.yaw)*cos(-img.pitch), -cos(img.roll)*sin(-img.pitch), -sin(img.roll)*sin(img.yaw)*sin(-img.pitch)+cos(img.yaw)*cos(-img.pitch))
	);

	// TODO: Turn this whole thing into a matrix operation or two
	int threadCount = std::thread::hardware_concurrency();
	std::thread *threads = new std::thread[threadCount];
	for (int n = 0; n < threadCount; n++) {
		threads[n] = std::thread([&, n]() {
			for (int y = n, yFlip = height - 1 - n; y < height; y += threadCount, yFlip -= threadCount) {
				unsigned char *outPtr = imgOut[y];
				for (int x = 0; x < width; x++) {
					int xFilter[2] = { std::min(x, width - 1), std::min(x + 1, width - 1) };
					int yFilter[2] = { std::min(yFlip, height - 1), std::min(yFlip + 1, height - 1) };
					unsigned char Rval[4];
					unsigned char Gval[4];
					unsigned char Bval[4];

					for (int a = 0; a < 2; a++) {
						for (int b = 0; b < 2; b++) {
							// Normalize x/y to (0,1) and convert into lat/long radians
							glm::vec2 LLA = xy_to_lla * glm::vec2(xFilter[a], yFilter[b]) - glm::vec2(M_PI, M_PI_2);
							// Convert LLA to spherical XYZ, then rotate
							glm::vec3 XYZ = rotate * glm::vec3(cos(LLA.y)*sin(LLA.x), sin(LLA.y), cos(LLA.y)*cos(LLA.x));
							// Back to LLA, undo normalization
							LLA = glm::vec2(
								((atan2(XYZ.x, XYZ.z) + M_PI) / (2.0 * M_PI))*width,
								((asin(XYZ.y) + M_PI_2) / M_PI)*height
							);
							// Round to nearest pixel
							int xi = (int)round(LLA.x);
							int yi = std::min((int)round(LLA.y), height-1);

							Rval[a * 2 + b] = img.texture.data[(xi + yi * width) * channels + 0];
							Gval[a * 2 + b] = img.texture.data[(xi + yi * width) * channels + 1];
							Bval[a * 2 + b] = img.texture.data[(xi + yi * width) * channels + 2];
						}
					}
					float weight[3] = { 0.5f, 0.5f, 0.5f };
					*(outPtr)++ = Bilinear(weight, Rval);
					*(outPtr)++ = Bilinear(weight, Gval);
					*(outPtr)++ = Bilinear(weight, Bval);
					
					//*(outPtr++) = img.texture.data[(xi + yi * width) * channels + 0];
					//*(outPtr++) = img.texture.data[(xi + yi * width) * channels + 1];
					//*(outPtr++) = img.texture.data[(xi + yi * width) * channels + 2];
				}
			}
		});
	}

	for (int n = 0; n < threadCount; n++) {
		threads[n].join();
	}
}

#endif //__IMAGEHANDLER_H