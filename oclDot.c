#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "cl-helper.h"


/* ---------------- Macros ---------------- */
const int VSIZE = 1024*1024*256;
const int LDIM = 256;


int32_t main(int32_t argc, char** argv) {
    int32_t hV1[VSIZE];
    int32_t hV2[VSIZE];
    int64_t hRES[VSIZE/LDIM];

    clock_t start, end;
    int64_t checksum = 0;
    int64_t res_sum = 0;

    // OpenCL variables
    cl_context ctx;
    cl_command_queue queue;
    cl_int status;

    size_t ldim[] = { LDIM };
    size_t gdim[] = {VSIZE};
    size_t ngr = VSIZE/LDIM;

    for (uint32_t i = 0; i < VSIZE; i++) {
        hV1[i] = (i+1);
        hV2[i] = 2;
        checksum += (i+1)*2;
    }



    /* ---------------- Requesting the device to run the computations ---------------- */

    create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);
    print_device_info_from_queue(queue);
    // Creating the kernel from file
    start = clock();
    char *knl_text = read_file("my_dot.cl");
    cl_kernel knl = kernel_from_string(ctx, knl_text, "my_dot", NULL);

    /* ---------------- Memory preallocation ---------------- */

    // Allocating memory on the device and copying the first vector
    cl_mem dV1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, VSIZE*sizeof(int32_t), 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, dV1, CL_TRUE,  0,
        VSIZE*sizeof(int32_t), hV1,
        0, NULL, NULL));

    // Allocating memory on the device and copying the second vector
    cl_mem dV2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, VSIZE*sizeof(int32_t), 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, dV2, CL_TRUE,  0,
        VSIZE*sizeof(int32_t), hV2,
        0, NULL, NULL));

    // Allocating memory for the result vector
    cl_mem dRES = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int64_t)*ngr, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    CALL_CL_GUARDED(clFinish, (queue));

    /* ---------------- Performing compputation and retrieving the result to hRES ---------------- */


    clSetKernelArg(knl, 0, sizeof(cl_mem), &dV1);
    clSetKernelArg(knl, 1, sizeof(cl_mem), &dV2);
    clSetKernelArg(knl, 2, sizeof(cl_mem), &dRES);
    clSetKernelArg(knl, 3, LDIM * sizeof(int64_t), NULL);
    clSetKernelArg(knl, 4, sizeof(int), &VSIZE);


    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
        (queue, knl,
         1, NULL, gdim, ldim,
         0, NULL, NULL));

    CALL_CL_GUARDED(clFinish, (queue));

    CALL_CL_GUARDED(clEnqueueReadBuffer, (
        queue, dRES, CL_TRUE,  0,
        sizeof(int64_t)*ngr, hRES,
        0, NULL, NULL));

    res_sum = 0;
    for (int i=0; i < ngr; i++)
        res_sum += hRES[i];

    end = clock();
    printf("Elapsed time for calculation (OpenCL): %.2f s.\n=============================\n", (double) (end - start) / CLOCKS_PER_SEC);

    /* ---------------- Checking the results ---------------- */

    printf("Result: %ld\n", res_sum);
    printf("Checksum: %ld\n", checksum);

    if(res_sum == checksum)
        printf("=============================\nSuccessful OpenCL computation!\n");


    /* ---------------- Cleaning ---------------- */

    CALL_CL_GUARDED(clReleaseMemObject, (dV1));
    CALL_CL_GUARDED(clReleaseMemObject, (dV2));
    CALL_CL_GUARDED(clReleaseMemObject, (dRES));
    CALL_CL_GUARDED(clReleaseKernel, (knl));
    CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
    CALL_CL_GUARDED(clReleaseContext, (ctx));

	return 0;
}
