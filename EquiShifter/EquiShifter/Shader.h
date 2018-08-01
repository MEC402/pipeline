#ifndef __SHADER_H
#define __SHADER_H

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <glm\gtc\matrix_transform.hpp>
#include "Quad.h"

class Shader {
public:
	Shader() = default;
	~Shader() = default;

	GLuint CreateProgram(void);
	void SetMVP(glm::mat4 MVP);
	void SetUniform(float value, const char *uniformName);
	void BindTexture(GLuint txID);
	void CreateTexture(Quad *q);

private:

	GLuint createShader(GLenum type, const GLchar *src);

	GLuint m_program;
	GLuint m_vertShader;
	GLuint m_fragShader;

};

#endif //__SHADER_H