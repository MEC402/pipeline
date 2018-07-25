#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>
#include <unistd.h>

// CImg library
#include "CImg.h"
using namespace cimg_library;

// Multithreading
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
using namespace tbb;

// Input parameters
int iflag, oflag, hflag, rflag;
char *ivalue, *ovalue;
int rvalue=4096;

struct PixelRange{ int start, end;};

// Overload for nesting Interpolate calls
inline unsigned char LinearInterpolate(float weight, unsigned char v1, unsigned char v2)
{
	return (unsigned char)(weight*v2 + (1.0f - weight) * v1);
}

// Expectes 1 weight and 2 values
inline unsigned char LinearInterpolate(float weight, unsigned char *values)
{
	return (unsigned char)(weight * (values[1]) + (1.0f - weight)*values[0]);
}

// Expects 2 weights and 4 values
inline unsigned char BilinearInterpolate(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		LinearInterpolate(weight[1], &values[0]),
		LinearInterpolate(weight[1], &values[2])
	};
	return LinearInterpolate(weight[0], prime);
}

// Expects 3 weights and 8 values
inline unsigned char TrilinearInterpolate(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		BilinearInterpolate(&(weight[0]), &(values[0])),
		BilinearInterpolate(&(weight[1]), &(values[4]))
	};
	return LinearInterpolate(weight[3], prime);
}

int parseParameters(int argc, char *argv[]) {
    iflag = oflag = hflag = rflag = 0;
    ivalue = ovalue = NULL;
    int c;
    opterr = 0;
    
    while ((c = getopt (argc, argv, "i:o:r:")) != -1)
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
                rvalue = std::stoi(optarg);
                break;
            case '?':
                if (optopt == 'i' || optopt == 'o' || optopt == 'r')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                return 1;
            default:
                abort ();
        }
    
    if (iflag==0 || oflag == 0) {
        std::cout << "No inputs or outputs specified: "<< iflag << "/" << oflag <<"\n";
        abort ();
        return 1;
    }
    return 0;
}

void outImgToXYZ(int, int, int, int, double*, double*, double*);
void convertBack(CImg<unsigned char>&, CImg<unsigned char> **);

int main (int argc, char *argv[]) {
    std::cout << "PeakVisor panorama translator...\n";
    
    parseParameters(argc, argv);
    
    std::cout << "  convert equirectangular panorama: [" << ivalue << "] into cube faces: ["<< ovalue << "] of " << rvalue <<" pixels in dimension\n";
    
    // Input image
    CImg<unsigned char> imgIn(ivalue);
    
    // Create output images
    CImg<unsigned char>* imgOut[6];
    for (int i=0; i<6; ++i){
        imgOut[i] = new CImg<unsigned char>(rvalue, rvalue, 1, 4, 255);
    }
    
    // Convert panorama
    convertBack(imgIn, imgOut);
    
    // Write output images
    for (int i=0; i<6; ++i){
        std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";//".jpg";
        imgOut[i]->save_png(fname.c_str());
    }
    
    std::cout << "  convertation finished successfully\n";
    return 0;
}



/**
 **	Convert panorama using an inverse pixel transformation
 **/
void convertBack(CImg<unsigned char>& imgIn, CImg<unsigned char> **imgOut){
    int _dw = rvalue*4;
    int edge = rvalue; // the length of each edge in pixels
    int face = 0;
    int width = imgIn.width();
    int height = imgIn.height();
    // Look around cube faces
    tbb::parallel_for(blocked_range<size_t>(0, _dw, 1), [&](const blocked_range<size_t>& range) {
	int face = 0;
        for (size_t i=range.begin(); i<range.end(); ++i) {
            face = int(i/edge); // 0 - back, 1 - left 2 - front, 3 - right
            PixelRange rng = {edge, 2*edge};
            
            if (i>=2*edge && i<3*edge) {
                rng = {0, 3*edge};
            }
            
            for (int j=rng.start; j<rng.end; ++j) {
                if (j<edge) {
                    face = 4;
                } else if (j>2*edge) {
                    face = 5;
                } else {
                    face = int(i/edge);
                }
		double x,y,z;
		outImgToXYZ(i, j, face, edge, &x, &y, &z);

                double theta = atan2(y, x);
		double r = hypot(x, y);
		double phi = atan2(z, r);
		double uf = (theta + M_PI) / M_PI * height;
		double vf = (M_PI_2 - phi) / M_PI * height;
		int ui = std::min(static_cast<int>(std::floor(uf)), width);
		int vi = std::min(static_cast<int>(std::floor(vf)), height);
		int u2 = std::min(ui + 1, width);
		int v2 = std::min(vi + 1, height);
		int u3 = std::min(ui + 2, width);
		int v3 = std::min(vi + 2, height-1);
		int u4 = std::max(ui - 1, 0);
		int v4 = std::max(vi - 1, 0);
		int u[4] = { ui, u2, u3, u4 };
		int v[4] = { vi, v2, v3, v4 };
	
		unsigned char Rval[16];
		unsigned char Gval[16];
		unsigned char Bval[16];
		unsigned char *data = imgIn.data();
		for (int a = 0; a < 4; a++) {
			for (int b = 0; b < 4; b++) {
				Rval[a * 4 + b] = data[u[a] + v[b] * width + 0 * width*height];
				Gval[a * 4 + b] = data[u[a] + v[b] * width + 1 * width*height];
				Bval[a * 4 + b] = data[u[a] + v[b] * width + 2 * width*height];
			}
		}

		float weight[3] = { 0.5f, 0.5f, 0.5f };
		unsigned char R = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Rval[0]), TrilinearInterpolate(weight, &Rval[8]));
		unsigned char G = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Gval[0]), TrilinearInterpolate(weight, &Gval[8]));
		unsigned char B = LinearInterpolate(weight[0], TrilinearInterpolate(weight, &Bval[0]), TrilinearInterpolate(weight, &Bval[8]));

		// Based on T-Strip coordinates, mod to edge size and insert into appropriate face
		int idx = ((i%edge) + (j%edge)*edge);
		unsigned char *ptr = imgOut[face]->data();
		// CImg uses planar RGBA storage, hence n*edge*edge for each value
		ptr[idx + 0 * edge*edge] = R;
		ptr[idx + 1 * edge*edge] = G;
		ptr[idx + 2 * edge*edge] = B;
		ptr[idx + 3 * edge*edge] = 255;

            }
        }
    });

}

void outImgToXYZ(int i, int j, int face, int edge, double *x, double *y, double *z)
{
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


