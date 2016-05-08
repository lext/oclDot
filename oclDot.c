#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "cl-helper.h"


/* ---------------- Macros ---------------- */
const int VSIZE = 1024*512*512;
// This macros was found somewhere on stackoverflow and it is used to release the memory of
// several arrays
#define FREE_ALL(...) \
do { \
    int i=0;\
    void *pta[] = {__VA_ARGS__}; \
    for(i=0; i < sizeof(pta)/sizeof(void*); i++) \
    { \
        free(pta[i]); \
    }\
} while(0)



int32_t main(int32_t argc, char** argv) {

    int32_t hV1[VSIZE];
    int32_t hV2[VSIZE];
    int32_t hRES[VSIZE];
    clock_t start, end;

    // OpenCL variables
    cl_context ctx;
    cl_command_queue queue;
    cl_int status;

    for (uint32_t i = 0; i < VSIZE; i++) {
        hV1[i] = (i+1);
        hV2[i] = 2;
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
    cl_mem dRES = clCreateBuffer(ctx, CL_MEM_READ_WRITE, VSIZE*sizeof(int32_t), 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    CALL_CL_GUARDED(clFinish, (queue));

    /* ---------------- Performing compputation and retrieving teh result to hRES ---------------- */

    SET_4_KERNEL_ARGS(knl, dV1, dV2, dRES, VSIZE);

    size_t ldim[] = { 512 };
    size_t gdim[] = {VSIZE};

    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
        (queue, knl,
         1, NULL, gdim, ldim,
         0, NULL, NULL));

    CALL_CL_GUARDED(clFinish, (queue));

    CALL_CL_GUARDED(clEnqueueReadBuffer, (
        queue, dRES, CL_TRUE,  0,
        VSIZE*sizeof(int32_t), hRES,
        0, NULL, NULL));
    end = clock();
    printf("Elapsed time for calculation: %.2f s.\n", (double)(end - start) / CLOCKS_PER_SEC);

    /* ---------------- Checking the results ---------------- */

    for (uint32_t i = 0; i < VSIZE; i++) {
        if(hRES[i] != 2*(i+1)) {
            printf("BUG!!!!\n");
            printf("Should be %d, but %d\n", 2*i, hRES[i]);
            exit(1);

        }
    }


    printf("Successful computation!\n");

	return 0;
}
