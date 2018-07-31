#ifndef __CAMERA_H
#define __CAMERA_H

#include <glm\gtc\matrix_transform.hpp>

class Camera {
public:
	Camera(int w, int h);
	~Camera() = default;

	void Zoom(float FOVdelta);
	void SetResolution(int w, int h);
	void SetModelScale(int w, int h);
	void ShiftModelMatrix(float x, float y, float z);
	glm::mat4 GetMVP(void);

	int width{ 1280 };
	int height{ 800 };

private:
	float yFOV{ 45.0f };
	glm::mat4 Projection;
	glm::mat4 View;
	glm::mat4 Model;

	void setPerspective(void);
};

#endif //__CAMERA_H