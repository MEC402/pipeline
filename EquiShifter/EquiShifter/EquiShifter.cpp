#include <cstdio>
#include <string>
#include <GL\glew.h>
#include <GL\freeglut.h>

#include "ImageHandler.h"
#include "Shader.h"
#include "Camera.h"
#include "Filters.h"
#include "Quad.h"

Camera *camera;
Shader *shader;
Quad imgA;
Quad imgB;

void Display(void);
void Idle(void);
void ProcessKeys(unsigned char key, int x, int y);
void ProcessSpecialKeys(int key, int x, int y);
void MouseWheel(int button, int direction, int x, int y);
void Resize(int w, int h);

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "No input file detected, please provide a path to a png image to load\n");
		return -1;
	}

	const char *pathA = argv[1];
	const char *pathB = NULL;
	if (argc > 2)
		pathB = argv[2];
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(1280, 800); // Defaults to 1280 x 800 windowed
	glutCreateWindow("EquiShifter - An Equirectangular Reprojection Tool");
	GLenum initErr = glewInit();
	if (GLEW_OK != initErr) {
		fprintf(stderr, "Error %s\n", glewGetErrorString(initErr));
	}
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glutDisplayFunc(Display);
	glutIdleFunc(Idle);
	glutReshapeFunc(Resize);
	glutKeyboardFunc(ProcessKeys);
	glutSpecialFunc(ProcessSpecialKeys);
	glutMouseWheelFunc(MouseWheel);

	
	printf("Loading image...\n");
	imgA.type = SetImagetype(std::string(pathA));
	imgA.texture = ReadImage(imgA.type, pathA);
	if (imgA.texture.data == NULL) {
		printf("Unsupported filetype.  Exiting. \n");
		return -1;
	}
	printf("Image loaded from disk\n");


	// Generate quad
	glGenBuffers(1, &imgA.posBuf);
	glBindBuffer(GL_ARRAY_BUFFER, imgA.posBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	glGenBuffers(1, &imgA.txBuf);
	glBindBuffer(GL_ARRAY_BUFFER, imgA.txBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadUVs), quadUVs, GL_STATIC_DRAW);
	imgA.indices = 2 * 3;

	camera = new Camera(1280, 800);
	camera->SetModelScale(imgA.texture.width, imgA.texture.height);

	shader = new Shader();
	shader->CreateProgram();
	shader->CreateTexture(&imgA);

	if (pathB != NULL) {
		printf("Loading second image...\n");
		imgB.type = SetImagetype(std::string(pathB));
		imgB.texture = ReadImage(imgB.type, pathB);
		shader->CreateTexture(&imgB);
		printf("Second image loaded\n");

		glGenBuffers(1, &imgB.posBuf);
		glBindBuffer(GL_ARRAY_BUFFER, imgB.posBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
		glGenBuffers(1, &imgB.txBuf);
		glBindBuffer(GL_ARRAY_BUFFER, imgB.txBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadUVs), quadUVs, GL_STATIC_DRAW);
		imgB.indices = 2 * 3;
	}

	glutMainLoop();

	return 0;
}





/* ------------------ Begin OpenGL Callbacks ------------------ */

void DrawQuad(Quad q)
{
	shader->SetUniform(q.alpha, "alpha");
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
}

void Display()
{
	glClearColor(0.5, 0.5, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(0, 0, camera->width, camera->height);
	
	if (imgB.texture.data != NULL) {
		DrawQuad(imgB);
	}

	DrawQuad(imgA);
	glutSwapBuffers();
}

void Idle()
{
	Display();
}

void ProcessSpecialKeys(int key, int x, int y)
{
	static float scale = 0.01f;
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
	case GLUT_KEY_F5:
		SaveImage(imgA);
		break;
	}
}

void ProcessKeys(unsigned char key, int x, int y)
{
	static float scale = 0.1f;
	switch (key) {
	case 'w':
		imgA.pitch += scale;
		break;
	case 's':
		imgA.pitch -= scale;
		break;
	case 'a':
		imgA.yaw += scale;
		break;
	case 'd':
		imgA.yaw -= scale;
		break;
	case 'e':
		imgA.roll += scale;
		break;
	case 'q':
		imgA.roll -= scale;
		break;
	case '+':
		imgA.alpha += (scale / 10.0f);
		if (imgA.alpha > 1.0f)
			imgA.alpha = 1.0f;
		break;
	case '-':
		imgA.alpha -= (scale / 10.0f);
		if (imgA.alpha < 0.0f)
			imgA.alpha = 0.0f;
		break;
	}
	printf("Pitch: %f | Yaw: %f | Roll %f\n", imgA.pitch, imgA.yaw, imgA.roll);
}

void MouseWheel(int button, int direction, int x, int y)
{
	if (direction > 0)
		camera->Zoom(0.01f);
	else
		camera->Zoom(-0.01f);
}

void Resize(int w, int h)
{
	camera->SetResolution(w, h);
}
