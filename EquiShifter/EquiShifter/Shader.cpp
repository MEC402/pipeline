#include "Shader.h"
#include "ShaderSrc.h"
#include <stdio.h>

GLuint Shader::CreateProgram()
{
	m_program = glCreateProgram();

	m_vertShader = createShader(GL_VERTEX_SHADER, vertSource);
	glAttachShader(m_program, m_vertShader);

	m_fragShader = createShader(GL_FRAGMENT_SHADER, fragSource);
	glAttachShader(m_program, m_fragShader);

	glLinkProgram(m_program);
	glUseProgram(m_program);
	
	return m_program;
}

void Shader::SetMVP(glm::mat4 MVP)
{
	GLuint MatrixID = glGetUniformLocation(m_program, "MVP");
	if (MatrixID == -1)
		fprintf(stderr, "Error getting MVP uniform!\n");
	else
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, (float*)&(MVP));
}

void Shader::SetUniform(float value, const char *uniformName)
{
	GLuint UniformID = glGetUniformLocation(m_program, uniformName);
	if (UniformID == -1)
		fprintf(stderr, "Error getting %s uniform!\n", uniformName);
	else
		glUniform1f(UniformID, value);
}

void Shader::BindTexture(GLuint txID)
{
	GLuint TxUniform = glGetUniformLocation(m_program, "image");
	if (TxUniform == -1) {
		fprintf(stderr, "Error getting image sampler!\n");
	}
	else {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, txID);
		glUniform1i(TxUniform, 0);
	}
}

void Shader::CreateTexture(int width, int height, unsigned char *data, Quad *q)
{
	q->texture.width = width;
	q->texture.height = height;
	glGenTextures(1, &q->texture.id);
	glBindTexture(GL_TEXTURE_2D, q->texture.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, q->texture.width, q->texture.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
}

/* --------------- Begin Private Functions --------------- */

GLuint Shader::createShader(GLenum type, const GLchar *src)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);
	GLint isCompiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
	if (isCompiled == GL_FALSE) {
		GLint maxLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
		char *errorLog = (char*)malloc(sizeof(char)*maxLength);
		glGetShaderInfoLog(shader, maxLength, &maxLength, errorLog);
		fprintf(stderr, "Error compiling shader: %s\n", src);
		fprintf(stderr, "%s\n", errorLog);
		glDeleteShader(shader); // Don't leak the shader.
		return NULL;
	}
	return shader;
}