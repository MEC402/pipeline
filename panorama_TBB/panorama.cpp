#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>
#include <unistd.h>

#include <chrono>
#include <thread>

// CImg library
#define cimg_use_png
#include "CImg/CImg.h"
using namespace cimg_library;

//#include <xmmintrin.h>
//#include <pmmintrin.h>

// Multithreading
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
using namespace tbb;

// Input parameters
int iflag, oflag, hflag, rflag;
char *ivalue, *ovalue;
int rvalue=4096;

/**
 **	Parse input parameters
 **/
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

struct PixelRange {int start, end; };

/** get x,y,z coords from out image pixels coords
 **	i,j are pixel coords
 **	face is face number
 **	edge is edge length
 **/
void outImgToXYZ(int, int, int, int, double*, double*, double*);
unsigned char* interpolateXYZtoColor(double,double,double, CImg<unsigned char>&);
/**
 **	Convert panorama using an inverse pixel transformation
 **/
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
    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Convert panorama
    convertBack(imgIn, imgOut);
    
    printf("Time to convert: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());

    std::thread threads[6];
    // Write output images
    for (int i=0; i<6; ++i){
        threads[i] = std::thread([&]() {
	    std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";//".jpg";
            imgOut[i]->save_png(fname.c_str());
	});
    }
    for (int i = 0; i < 6; i++) {
        threads[i].join();
    }
    /*for (int i = 0; i < 6; i++) {
	std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".png";
	imgOut[i]->save_png(fname.c_str());
    }*/
    printf("Total time: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
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
    
    // Look around cube faces
	tbb::parallel_for(blocked_range<size_t>(0, _dw, 1), [&](const blocked_range<size_t>& range) {
		for (size_t i=range.begin(); i<range.end(); ++i) {
//		for (int i = 0; i < _dw; i++) {
			int face = int(i/edge);
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
               			double x, y, z; 
		                outImgToXYZ(i, j, face, edge, &x, &y, &z);
//		                unsigned char *color = interpolateXYZtoColor(x,y,z,imgIn);

				int _sw = imgIn.width();
				int _sh = imgIn.height();
				double theta = std::atan2(y, x);
				double r = std::hypot(x, y);
				double phi = std::atan2(z, r);
				double uf = (theta + M_PI) / M_PI * _sh;
				double vf = (M_PI_2 - phi) / M_PI * _sh; // implicit assumption: _sh == _sw / 2
				int ui = std::min(static_cast<int>(std::floor(uf)), _sw);
				int vi = std::min(static_cast<int>(std::floor(vf)), _sh);
				int u2 = std::min(ui+1, _sw);
				int v2 = std::min(vi+1, _sh);
				double mu = uf - ui, nu = vf - vi;
				mu = nu = 0;
				int width = _sw;
				int height = _sh;

				unsigned char Ra = imgIn[ui + vi * width + 0] + (imgIn[u2 + vi * width + 0] - imgIn[ui + vi * width + 0]) * mu;
				unsigned char Ga = imgIn[ui + vi * width + 1*width*height] + (imgIn[u2 + vi * width + 1*width*height] - imgIn[ui + vi * width + 1*width*height]) * mu;
				unsigned char Ba = imgIn[ui + vi * width + 2*width*height] + (imgIn[u2 + vi * width + 2*width*height] - imgIn[ui + vi * width + 2*width*height]) * mu;
				unsigned char Rb = imgIn[ui + v2 * width + 0] + (imgIn[u2 + v2 * width + 0] - imgIn[ui + v2 * width + 0]) * mu;
				unsigned char Gb = imgIn[ui + v2 * width + 1*width*height] + (imgIn[u2 + v2 * width + 1*width*height] - imgIn[ui + v2 * width + 1*width*height]) * mu;
				unsigned char Bb = imgIn[ui + v2 * width + 2*width*height] + (imgIn[u2 + v2 * width + 2*width*height] - imgIn[ui + v2 * width + 2*width*height]) * mu;
				unsigned char R = Ra + (Rb - Ra) * nu;
				unsigned char G = Ga + (Gb - Ga) * nu;
				unsigned char B = Ba + (Bb - Ba) * nu;

				int whd = imgOut[face]->width()*imgOut[face]->height();
				int idx = (i%edge) + (j%edge)*edge;
				unsigned char *ptr = imgOut[face]->data();
				ptr[idx + 0*edge*edge] = R;
				ptr[idx + 1*edge*edge] = G;
				ptr[idx + 2*edge*edge] = B;
				ptr[idx + 3*edge*edge] = 255;

/*				unsigned char *ptr = imgOut[face]->data((i%edge), (j%edge), 0, 0);
				int bound = imgOut[face]->spectrum();
				for (int k = 0; k < bound; k++) {
					*ptr = (unsigned char)(color[k]);
					ptr += whd;
				}*/
		     	}
		}
	});
}

void outImgToXYZ(int i, int j, int face, int edge, double *x, double *y, double *z) {
    auto a = 2.0 * i / edge;
    auto b = 2.0 * j / edge;

    if (face == 0) { // back
	*x = -1;
	*y = 1-a;
	*z = 3-b;
    } else if (face == 1) { // left
       	*x = a-3;
	*y = -1;
	*z = 3-b;
    } else if (face == 2) { // front
 	*x = 1;
	*y = a-5;
	*z = 3-b;
    } else if (face == 3) { // right
 	*x = 7-a;
	*y = 1;
	*z = 3-b;
    } else if (face == 4) { // top
 	*x = b-1;
	*y = a-5;
	*z = 1;
    } else if (face==5) { // bottom
 	*x = 5-b;
	*y = a-5;
	*z = -1;
    }
}

unsigned char* ReadColor(int x, int y, CImg<unsigned char>& imgIn)
{
	unsigned char *color = new unsigned char(4);
	unsigned char *data = imgIn.data();
	int width = imgIn.width();
	int height = imgIn.height();
	color[0] = data[x + y*width];
	color[1] = data[x + y*width + 1*width*height];
	color[2] = data[x + y*width + 2*width*height];
	color[3] = 255;
	return color;
}

unsigned char* MixColor(unsigned char* colorA, unsigned char* colorB, double weight)
{

	unsigned char *color = new unsigned char(4);
	color[0] = colorA[0] + (colorB[0] - colorA[0]) * weight;
	color[1] = colorA[1] + (colorB[1] - colorA[1]) * weight;
	color[2] = colorA[2] + (colorB[2] - colorA[2]) * weight;
	color[3] = 255;

	delete[]colorA;
	delete[]colorB;
	return color;
}

unsigned char* interpolateXYZtoColor(double x, double y, double z, CImg<unsigned char>& imgIn) {
	int _sw = imgIn.width();
	int _sh = imgIn.height();

	double theta = std::atan2(y, x);
	double r = std::hypot(x, y);// # range -pi to pi
	double phi = std::atan2(z, r);// # range -pi/2 to pi/2

	// source img coords
	double uf = (theta + M_PI) / M_PI * _sh;
	double vf = (M_PI_2 - phi) / M_PI * _sh; // implicit assumption: _sh == _sw / 2
	// Use bilinear interpolation between the four surrounding pixels
	int ui = std::min(static_cast<int>(std::floor(uf)), _sw);
	int vi = std::min(static_cast<int>(std::floor(vf)), _sh);
	int u2 = std::min(ui+1, _sw);
	int v2 = std::min(vi+1, _sh);
	double mu = uf - ui, nu = vf - vi;      //# fraction of way across pixel
	mu = nu = 0;
	auto A = ReadColor(ui,vi,imgIn), B = ReadColor(u2,vi,imgIn), C = ReadColor(ui,v2,imgIn), D = ReadColor(u2,v2,imgIn);
	auto value = MixColor(MixColor(A, B, mu), MixColor(C,D,mu), nu);

	return value;
}
