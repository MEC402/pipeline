#ifndef __QUAD_H
#define __QUAD_H

#include <GL\glew.h>

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
};

struct Quad {
	GLuint posBuf;
	GLuint txBuf;
	int indices;
	float yaw{ 0.0f };
	float pitch{ 0.0f };
	float roll{ 0.0f };
	Texture texture;
};

#endif //__QUAD_H