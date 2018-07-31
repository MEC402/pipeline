
// Overload for nesting Interpolate calls
inline unsigned char Linear(float weight, unsigned char v1, unsigned char v2)
{
	unsigned char result = weight * v2 + (1.0f - weight) * v1;
	return result;
}

// Expectes 1 weight and 2 values
inline unsigned char Linear(float weight, unsigned char *values)
{
	unsigned char result = (weight * values[1] + (1.0f - weight)*values[0]);
	return result;
	//return unsigned char(weight * (values[1]) + (1.0f - weight)*values[0]);
}

// Expects 2 weights and 4 values
inline unsigned char Bilinear(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		Linear(weight[1], &values[0]),
		Linear(weight[1], &values[2])
	};
	unsigned char result = Linear(weight[0], prime);
	return result;
}

// Expects 3 weights and 8 values
inline unsigned char Trilinear(float *weight, unsigned char *values)
{
	unsigned char prime[2] = {
		Bilinear(&(weight[0]), &(values[0])),
		Bilinear(&(weight[1]), &(values[4]))
	};
	unsigned char result = Linear(weight[2], prime);
	return result;
}