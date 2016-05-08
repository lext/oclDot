#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void my_dot(__global int *v1, __global  int *v2, __global int *res, uint N) {
	
	int i = get_global_id(0);

	if (i < N)
		res[i] = v1[i]*v2[i];

}