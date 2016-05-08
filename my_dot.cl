#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void my_dot(__global int *v1, __global  int *v2, __global long *res, __local long *tmp, uint N) {

	

	int i = get_global_id(0);
	int j = get_local_id(0);
	int gid = get_group_id(0);

	if (i < N) {
		// Multiplying numbers in local memory
		tmp[j] = v1[i]*v2[i];

		barrier(CLK_LOCAL_MEM_FENCE);

		// Doing reduction in local memory

		for(int offs=1; offs < 256; offs*= 2) {
			int mask = 2*offs - 1;
			barrier(CLK_LOCAL_MEM_FENCE);
			if (!(j&mask)) {
				tmp[j] += tmp[j+offs];
			}

		}

		barrier(CLK_LOCAL_MEM_FENCE);
		if(j == 0)
			res[gid] = tmp[0];	
	}

}