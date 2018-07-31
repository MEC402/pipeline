#define _USE_MATH_DEFINES
#define GLM_ENABLE_EXPERIMENTAL

#include <cmath>
#include <thread>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <GL\glew.h>
#include <GL\freeglut.h>

#include "png.h"
#include "Shader.h"
#include "Camera.h"
#include "Filters.h"
#include "Quad.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

Camera *camera;
Shader *shader;
Quad q;

void display(void);
void idle(void);
void ProcessKeys(unsigned char key, int x, int y);
void ProcessSpecialKeys(int key, int x, int y);
void MouseWheel(int button, int direction, int x, int y);
void Resize(int w, int h);

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE | GLUT_STENCIL);
	glutInitWindowSize(1280, 800); // Defaults to 1280 x 800 windowed
	glutCreateWindow("Pixel Picker");
	GLenum initErr = glewInit();
	if (GLEW_OK != initErr) {
		fprintf(stderr, "Error %s\n", glewGetErrorString(initErr));
	}

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(Resize);
	glutKeyboardFunc(ProcessKeys);
	glutSpecialFunc(ProcessSpecialKeys);
	glutMouseWheelFunc(MouseWheel);

	// Generate quad
	glGenBuffers(1, &q.posBuf);
	glBindBuffer(GL_ARRAY_BUFFER, q.posBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	glGenBuffers(1, &q.txBuf);
	glBindBuffer(GL_ARRAY_BUFFER, q.txBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadUVs), quadUVs, GL_STATIC_DRAW);
	q.indices = 2 * 3;

	printf("Loading image...\n");
	int width, height, channels;
	stbi_flip_vertically_on_write(true);
	stbi_set_flip_vertically_on_load(true);
	unsigned char *data = stbi_load("C:\\Users\\W8\\Desktop\\world.png", &width, &height, &channels, 0);
		
	camera = new Camera(1280, 800);
	camera->SetModelScale(width, height);

	shader = new Shader();
	shader->CreateProgram();
	shader->CreateTexture(width, height, data, &q);

	glutMainLoop();



	/*
	double LatScale = (360.0 / width);
	double LngScale = (180.0 / height);

	//CImg<unsigned char> ImgOut(width, height, 1, 3, 255);
	// Latitude -- (-180, 180)
	// Longitude -- (-90, 90)
	int threadCount = std::thread::hardware_concurrency();
	std::thread *threads = new std::thread[threadCount];

	unsigned char *InPtr = ImgIn.data();
	unsigned char *OutPtr = ImgOut.data();
	for (int n = 0; n < threadCount; n++) {
		threads[n] = std::thread([&, n]() {
			for (int x = n; x < width; x += threadCount) {
				for (int y = 0; y < height; y++) {
					double xx = 2 * (x + 0.5) / width - 1.0;
					double yy = 2 * (y + 0.5) / height - 1.0;
					double lng = M_PI * xx;
					double lat = M_PI_2 * yy;
					double X, Y, Z, D;
					int ix, iy;

					X = cos(lat) * cos(lng);
					Y = cos(lat) * sin(lng);
					Z = sin(lat);

					D = sqrt(X * X + Y * Y);

					glm::mat4 rot(1);
					rot = glm::rotate(rot, glm::radians(-150.0f), glm::vec3(0.0, 0.0, 1.0));
					glm::vec3 outXYZ = glm::vec3(rot * glm::vec4(X, Y, Z, 1.0));
					X = outXYZ.x;
					Y = outXYZ.y;
					Z = outXYZ.z;

					lat = atan2(Z, D);
					lng = atan2(Y, X);

					ix = (0.5 * lng / M_PI + 0.5) * width - 0.5;
					iy = (lat / M_PI + 0.5) * height - 0.5;

					int ui = std::min(ix, width-1);
					int vi = std::min(iy, height-1);
					int u2 = std::min(ui + 1, width-1);
					int v2 = std::min(vi + 1, height-1);
					int u3 = std::min(ui + 2, width-1);
					int v3 = std::min(vi + 2, height-1);
					int u4 = std::max(ui - 1, 0);
					int v4 = std::max(vi - 1, 0);
					int u[4] = { ui, u2, u3, u4 };
					int v[4] = { vi, v2, v3, v4 };

					unsigned char Rval[16];
					unsigned char Gval[16];
					unsigned char Bval[16];

					for (int a = 0; a < 4; a++) {
						for (int b = 0; b < 4; b++) {
							Rval[a * 2 + b] = InPtr[u[a] + v[b] * width + 0 * width*height];
							Gval[a * 2 + b] = InPtr[u[a] + v[b] * width + 1 * width*height];
							Bval[a * 2 + b] = InPtr[u[a] + v[b] * width + 2 * width*height];
						}
					}

					unsigned char R = Rval[0];
					unsigned char G = Gval[0];
					unsigned char B = Bval[0];
					//float weight[3] = { 0.5f, 0.5f, 0.5f };
					//unsigned char R = Bilinear(weight, Rval);
					//unsigned char G = Bilinear(weight, Gval);
					//unsigned char B = Bilinear(weight, Bval);
					//unsigned char R = Linear(weight[0], Trilinear(weight, &Rval[0]), Trilinear(weight, &Rval[8]));
					//unsigned char G = Linear(weight[0], Trilinear(weight, &Gval[0]), Trilinear(weight, &Gval[8]));
					//unsigned char B = Linear(weight[0], Trilinear(weight, &Bval[0]), Trilinear(weight, &Bval[8]));



					OutPtr[x + y * width + 0 * width*height] = R;
					OutPtr[x + y * width + 1 * width*height] = G;
					OutPtr[x + y * width + 2 * width*height] = B;
				}
			}
		});
	}
	
	for (int n = 0; n < threadCount; n++) {
		threads[n].join();
	}
	
	printf("Saving...\n");
	ImgOut.save_png("Out.png");
	*/
	return 0;
}

/* ------------------ Begin OpenGL Callbacks ------------------ */

void display()
{
	glClearColor(0.5, 0.5, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(0, 0, camera->width, camera->height);
	
	shader->SetUniform(q.yaw, "yaw");
	shader->SetUniform(q.pitch, "pitch");
	shader->SetUniform(q.roll, "roll");
	shader->BindTexture(q.texture.id);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, q.posBuf);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, q.txBuf);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

	shader->SetMVP(camera->GetMVP());
	glDrawArrays(GL_TRIANGLES, 0, q.indices);

	glutSwapBuffers();
}

void idle()
{
	display();
}

void ProcessSpecialKeys(int key, int x, int y)
{
	static float scale = 0.1f;
	switch (key) {
	case GLUT_KEY_LEFT:
		camera->ShiftModelMatrix(-scale, 0, 0);
		break;
	case GLUT_KEY_RIGHT:
		camera->ShiftModelMatrix(scale, 0, 0);
		break;
	case GLUT_KEY_DOWN:
		camera->ShiftModelMatrix(0, -scale, 0);
		break;
	case GLUT_KEY_UP:
		camera->ShiftModelMatrix(0, scale, 0);
		break;
	}
}

void ProcessKeys(unsigned char key, int x, int y)
{
	static float scale = 0.1f;
	switch (key) {
	case 'w':
		q.pitch += scale;
		break;
	case 's':
		q.pitch -= scale;
		break;
	case 'a':
		q.yaw += scale;
		break;
	case 'd':
		q.yaw -= scale;
		break;
	case 'e':
		q.roll += scale;
		break;
	case 'q':
		q.roll -= scale;
		break;
	}
}

void MouseWheel(int button, int direction, int x, int y)
{
	if (direction > 0)
		camera->Zoom(0.1f);
	else
		camera->Zoom(-0.1f);
}

void Resize(int w, int h)
{
	camera->SetResolution(w, h);
}
