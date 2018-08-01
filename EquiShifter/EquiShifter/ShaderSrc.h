/*
	GLSL source, eliminates the need to ship with separate shader files
*/

#ifndef __SHADERSRC_H
#define __SHADERSRC_H

#include <GL\glew.h>

const GLchar *vertSource = R"END(
#version 330 core
layout(location = 0) in vec3 vertPos;
layout(location = 1) in vec2 txCoord;

out vec2 f_txCoord;

uniform mat4 MVP;
uniform int width;
uniform int height;

void main()
{
	gl_Position = MVP * vec4(vertPos, 1.0);
	f_txCoord = txCoord;
}
)END";

const GLchar *fragSource = R"END(
#version 330 core
uniform sampler2D image;

uniform float alpha;
uniform float yaw;
uniform float pitch;
uniform float roll;

in vec2 f_txCoord;
out vec4 outColor;

vec3 rotateX(vec3 p, float theta)
{
	vec3 q;
	q.x = p.x;
	q.y = p.y * cos(theta) + p.z * sin(theta);
	q.z = -p.y * sin(theta) + p.z * cos(theta);
	return q;
}

vec3 rotateY(vec3 p, float theta)
{
	vec3 q;
	q.x = p.x * cos(theta) - p.z * sin(theta);
	q.y = p.y;
	q.z = p.x * sin(theta) + p.z * cos(theta);
	return q;
}

vec3 rotateZ(vec3 p, float theta)
{
	vec3 q;
	q.x = p.x * cos(theta) + p.y * sin(theta);
	q.y = -p.x * sin(theta) + p.y * cos(theta);
	q.z = p.z;
	return q;
}

float PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;

void main()
{
	float lat = f_txCoord.y * PI - PI/2.0;
	float lng = f_txCoord.x * 2.0 * PI - PI;
	vec3 ray;
	ray.x = cos(lat) * sin(lng);
	ray.y = sin(lat);
	ray.z = cos(lat) * cos(lng);

	ray = rotateX(ray, pitch);
	ray = rotateY(ray, yaw);
	ray = rotateZ(ray, roll);

	lat = asin(ray.y);
	lng = atan(ray.x, ray.z); //Calls atan2 function
	
	vec2 txCoords;
	txCoords.x = (lng + PI)/(2.0*PI);
	txCoords.y = (lat + PI/2.0)/PI;


	outColor = vec4(texture(image, txCoords).rgb, alpha);
}
)END";

#endif // __SHADERSRC_H