#include "Camera.h"

Camera::Camera(int w, int h) : 
	width(w), height(h)
{
	setPerspective();
	View = glm::lookAt(
		glm::vec3(0, 0, 1),
		glm::vec3(0, 0, 0),
		glm::vec3(0, 1, 0)
	);
	Model = glm::mat4(1.0f);
	Model = glm::translate(Model, glm::vec3(0, 0, -1));
}

void Camera::Zoom(float FOVdelta)
{
	yFOV += FOVdelta;
	setPerspective();
}

void Camera::SetResolution(int w, int h)
{
	width = w;
	height = h;
	setPerspective();
}

void Camera::SetModelScale(int w, int h)
{
	float aspect = float(w) / h;
	Model = glm::scale(Model, glm::vec3(aspect, 1, 1));
}

void Camera::ShiftModelMatrix(float x, float y, float z)
{
	Model = glm::translate(Model, glm::vec3(x, y, z));
}

glm::mat4 Camera::GetMVP()
{
	return Projection * View * Model;
}

void Camera::setPerspective()
{
	Projection = glm::perspective(yFOV, float(width) / height, 0.1f, 1000.0f);
}