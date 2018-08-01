#ifndef __QUAD_H
#define __QUAD_H

#include <GL\glew.h>
enum IMGenum { PNG = 0, JPG, TIF };

const float quadVertices[18] = {
	// Triangle 1
	-1, -1, 0,
	1, -1, 0,
	1, 1, 0,
	// Triangle 2
	1, 1, 0,
	-1, 1, 0,
	-1, -1, 0
};
const float quadUVs[12] = {
	// Triangle 1
	0, 0,
	1, 0,
	1, 1,
	// Triangle 2
	1, 1,
	0, 1,
	0, 0
};

struct Texture {
	GLuint id;
	unsigned char *data = NULL;
	int width;
	int height;
	int channels;
};

struct Quad {
	GLuint posBuf;
	GLuint txBuf;
	IMGenum type;
	int indices;
	float yaw{ 0.0f };
	float pitch{ 0.0f };
	float roll{ 0.0f };
	float alpha{ 1.0f };
	Texture texture;
};

#endif //__QUAD_H