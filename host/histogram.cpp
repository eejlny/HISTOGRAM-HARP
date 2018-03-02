/* File: histogram.cpp
 *
 Copyright (c) [2016] [Mohammad Hosseinabady (mohammad@hosseinabady.com)]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
* This file has been written at University of Bristol
* for the ENPOWER project funded by EPSRC
*
* File name : histogram.cpp
* author    : Mohammad hosseinabady mohammad@hosseinabady.com
* date      : 1 October 2017
* blog: https://highlevel-synthesis.com/
*/
#include "histogram.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>


#define GLOBAL_SIZE(x)	1;
double getTimestamp();
int load_file_to_memory(const char *filename, char **result);
void histogram_golden(INPUT_DATA_TYPE *Data, BIN_DATA_TYPE *Histogram, int data_size, int bin_size);


double start_app_time;
double end_app_time;
double app_total_time;



int main(int argc, char** argv) {


	int data_size = DATA_LENGTH;
	int bin_size = BIN_SIZE;

	printf("From main: Hello Histogram Version:01 \n");
	printf("From main: =====================\n");





	INPUT_DATA_TYPE *h_Data;
	BIN_DATA_TYPE *h_Histogram;
	BIN_DATA_TYPE *h_Histogram_golden;




	int err;
    size_t global[1];                   // global domain size
    size_t local[1];                    // local domain size

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;      // compute programs

    cl_kernel read_kernel;                   // compute mean kernel
    cl_kernel compute_histogram_kernel;                   // compute reduce kernel



    cl_mem d_Data;                         // device memory used for data
    cl_mem d_Histogram;                         // device memory used for mean


    cl_mem_ext_ptr_t d_Data_ext;
    cl_mem_ext_ptr_t d_Histogram_ext;

    cl_ulong time_start, time_end;
    double total_time;


    h_Data = (INPUT_DATA_TYPE*)malloc(sizeof(INPUT_DATA_TYPE)*data_size);
    h_Histogram = (BIN_DATA_TYPE*)malloc(sizeof(BIN_DATA_TYPE)*bin_size);
    h_Histogram_golden = (BIN_DATA_TYPE*)malloc(sizeof(BIN_DATA_TYPE)*bin_size);



    //initialization

    for(int i = 0; i < data_size; i++) {

    	BIN_DATA_TYPE t;

    	t = 1;//rand()%bin_size;
    	h_Data[i] = t;

    }

    for(int i = 0; i < bin_size; i++) {
   		h_Histogram[i] = 0;
   		h_Histogram_golden[i] = 0;
    }



    d_Data_ext.flags = XCL_MEM_DDR_BANK0;
    d_Data_ext.obj   = h_Data;
    d_Data_ext.param = 0;


    d_Histogram_ext.flags = XCL_MEM_DDR_BANK0;
    d_Histogram_ext.obj   = h_Histogram;
    d_Histogram_ext.param = 0;

	 // Connect to first platform
	 //
	 err = clGetPlatformIDs(1,&platform_id,NULL);
	 if (err != CL_SUCCESS) {
		 printf("Error: Failed to find an OpenCL platform!\n");
	     printf("Test failed\n");
	     return EXIT_FAILURE;
	 }
		{
			int num_platforms = 1;
			char buffer[10240];
			printf(" %d platform(s) found\n", num_platforms);
			printf(" =====================\n");
			printf("\n");

			for (int i = 0; i <num_platforms; i++) {
				printf("platform number %d \n", i);
				printf("------------------------\n");
				clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, 10240, buffer, NULL);
				printf("  CL_PLATFORM_PROFILE = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 10240, buffer, NULL);
				printf("  CL_PLATFORM_VERSION = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 10240, buffer, NULL);
				printf("  CL_PLATFORM_NAME = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 10240, buffer, NULL);
				printf("  CL_PLATFORM_VENDOR = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
				printf("  CL_PLATFORM_EXTENSIONS = %s\n", buffer);



			}
			printf("\n");
			printf("\n");
			printf("\n");
		}


	  // Connect to a compute device
	    //
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
	                         1, &device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}


	{
		char     buffer[10240];
		cl_ulong buf_ulong;
		cl_uint  buf_uint;
		size_t   buf_size_arr[3];
		size_t   buf_size;

		printf(" 1 device found\n");
		printf(" =====================\n");
		printf("\n");
		printf("------------------------\n");
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, 10240, buffer, NULL);
		printf("  CL_DEVICE_NAME = %s\n", buffer);

		clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 10240, buffer, NULL);
		printf("  CL_DEVICE_VENDOR = %s\n", buffer);

		clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
		printf("  CL_DEVICE_MAX_COMPUTE_UNITS = %u\n",  buf_uint);

		clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
		printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY = %u\n",  buf_uint);

		clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  CL_DEVICE_GLOBAL_MEM_SIZE = %lu\n",  buf_ulong);

		clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  CL_DEVICE_LOCAL_MEM_SIZE = %lu\n",  buf_ulong);

		clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES , sizeof(buf_size_arr), buf_size_arr, NULL);
		printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES = %lu/%lu/%lu \n", buf_size_arr[0], buf_size_arr[1], buf_size_arr[2]);

		clGetDeviceInfo(device_id,  CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(buf_size), &buf_size, NULL);
		printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE = %lu \n", buf_size);

		printf("\n");
		printf("\n");
		printf("\n");
	}
	// Create a compute context
	//
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		printf("Error: Failed to create a compute context!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Create a command commands
	//
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
	if (!commands) {
		printf("Error: Failed to create a command commands!\n");
	    printf("Error: code %i\n",err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	int status;


	// Load binary from disk

	unsigned char* kernelbinary;
	char *xclbin = argv[1];


	//------------------------------------------------------------------------------
	// xclbin mean
	//------------------------------------------------------------------------------
	printf("INFO: loading xclbin_mean %s\n", xclbin);
	int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
	if (n_i < 0) {
		printf("failed to load kernel from xclbin_mean: %s\n", xclbin);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	size_t n0 = n_i;

	// Create the compute program from offline
	program = clCreateProgramWithBinary(context, 1, &device_id, &n0,
	                                        (const unsigned char **) &kernelbinary, &status, &err);

	if ((!program) || (err!=CL_SUCCESS)) {
		printf("Error: Failed to create compute program0 from binary %d!\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Build the program executable
	//
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
	    char buffer[2048];

	    printf("Error: Failed to build program executable!\n");
	    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	    printf("%s\n", buffer);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Create the compute kernel in the program we wish to run
	//
	read_kernel = clCreateKernel(program, "read_data_kernel", &err);
	if (!read_kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create read_data_kernel!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	compute_histogram_kernel = clCreateKernel(program, "compute_data_histogram_kernel", &err);
	if (!compute_histogram_kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create compute_histogram_kernel!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}


    //------------------------------------------------------------------------------





    // Create the input and output arrays in device memory for our calculation
	//
	d_Data = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_COPY_HOST_PTR,  sizeof(INPUT_DATA_TYPE) * data_size, &d_Data_ext, NULL);
	d_Histogram = clCreateBuffer(context,  CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_COPY_HOST_PTR, sizeof(BIN_DATA_TYPE) * bin_size, &d_Histogram_ext, NULL);

	if (!d_Data || !d_Histogram) {
		printf("Error: Failed to allocate device memory!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}



	// Write our data set into the input array in device memory
	//
/*	cl_event transfer_a_event;
	err = clEnqueueWriteBuffer(commands, d_Data, CL_TRUE, 0, sizeof(DATA_TYPE) * data_size, h_Data, 0, NULL, &transfer_a_event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to write to source array a!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}
	clFinish(commands);
*/
	// Set the arguments to our mean kernel
	//
	start_app_time=getTimestamp();
	err = 0;
	err  = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &d_Data);
	err  |= clSetKernelArg(read_kernel, 1, sizeof(int), &data_size);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}


	cl_event read_kernel_event;
	local[0]  = 1;
	global[0] = 1;
	err = clEnqueueNDRangeKernel(commands, read_kernel, 1, NULL,
	                                   (size_t*)&global, (size_t*)&local, 0, NULL, &read_kernel_event);
	if (err) {
		printf("Error: Failed to execute kernel! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}




	// Set the arguments to our reduce kernel
	//
	err = 0;
	err   = clSetKernelArg(compute_histogram_kernel, 0, sizeof(int), &data_size);
	err  |= clSetKernelArg(compute_histogram_kernel, 1, sizeof(int), &bin_size);
	err  |= clSetKernelArg(compute_histogram_kernel, 2, sizeof(cl_mem), &d_Histogram);

    if (err != CL_SUCCESS) {
    	printf("Error: Failed to set reduce kernel arguments! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}

    cl_event compute_histogram_kernel_event;
	local[0]  = 1;
	global[0] = 1;
	err = clEnqueueNDRangeKernel(commands, compute_histogram_kernel, 1, NULL,
		                                   (size_t*)&global, (size_t*)&local, 0, NULL, &compute_histogram_kernel_event);

	if (err) {
		printf("Error: Failed to execute kernel! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}


   	clFinish(commands);



	// Read back the results from the device to verify the output
	//

	cl_event transfer_c_event;
	err = clEnqueueReadBuffer( commands, d_Histogram, CL_TRUE, 0, sizeof(BIN_DATA_TYPE) * bin_size, h_Histogram, 0, NULL, &transfer_c_event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}
   	clFlush(commands);
   	clFinish(commands);

   	end_app_time=getTimestamp();


   	app_total_time = (end_app_time-start_app_time)/1000;
   	printf("First App total execution time  %.6lf ms elapsed\n", app_total_time);


	// Set the arguments to our mean kernel
	//
	start_app_time=getTimestamp();
	err = 0;
	err  = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &d_Data);
	err  |= clSetKernelArg(read_kernel, 1, sizeof(int), &data_size);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}



	local[0]  = 1;
	global[0] = 1;
	err = clEnqueueNDRangeKernel(commands, read_kernel, 1, NULL,
	                                   (size_t*)&global, (size_t*)&local, 0, NULL, &read_kernel_event);
	if (err) {
		printf("Error: Failed to execute kernel! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}




	// Set the arguments to our reduce kernel
	//
	err = 0;
	err   = clSetKernelArg(compute_histogram_kernel, 0, sizeof(int), &data_size);
	err  |= clSetKernelArg(compute_histogram_kernel, 1, sizeof(int), &bin_size);
	err  |= clSetKernelArg(compute_histogram_kernel, 2, sizeof(cl_mem), &d_Histogram);

    if (err != CL_SUCCESS) {
    	printf("Error: Failed to set reduce kernel arguments! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}


	local[0]  = 1;
	global[0] = 1;
	err = clEnqueueNDRangeKernel(commands, compute_histogram_kernel, 1, NULL,
		                                   (size_t*)&global, (size_t*)&local, 0, NULL, &compute_histogram_kernel_event);

	if (err) {
		printf("Error: Failed to execute kernel! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}


   	clFinish(commands);



	// Read back the results from the device to verify the output
	//


	err = clEnqueueReadBuffer( commands, d_Histogram, CL_TRUE, 0, sizeof(BIN_DATA_TYPE) * bin_size, h_Histogram, 0, NULL, &transfer_c_event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}
   	clFlush(commands);
   	clFinish(commands);

   	end_app_time=getTimestamp();


   	app_total_time = (end_app_time-start_app_time)/1000;
   	printf("Second App total execution time  %.6lf ms elapsed\n", app_total_time);


	clGetEventProfilingInfo(read_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(read_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for read kernel in milliseconds = %0.3f ms\n", (total_time / 1000000.0));


	clGetEventProfilingInfo(compute_histogram_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(compute_histogram_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for add kernel in milliseconds = %0.3f ms\n", (total_time / 1000000.0));


	clGetEventProfilingInfo(transfer_c_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(transfer_c_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for transfer c in milliseconds = %0.3f ms\n", (total_time / 1000000.0));

	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Histogram);
	clReleaseKernel(read_kernel);
	clReleaseKernel(compute_histogram_kernel);
	clReleaseEvent(transfer_c_event);
	clReleaseEvent(compute_histogram_kernel_event);
	clReleaseEvent(read_kernel_event);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	clReleaseProgram(program);
	clReleaseDevice(device_id);


    histogram_golden(h_Data, h_Histogram_golden, data_size, bin_size);

    for (int i = 0; i < bin_size; i++) {
    	BIN_DATA_TYPE gold=h_Histogram_golden[i];
	   	BIN_DATA_TYPE hw = h_Histogram[i];
	    BIN_DATA_TYPE diff = (gold-hw);
	    if (diff != 0) {
	    	printf("Error at element %d golden= %d, hw=%d\n", i, gold, hw);
//	    	break;
	    }
	}

    printf("From main: Bye Histogram\n");
    printf("From main: ====================\n");
    return 0;
}

void histogram_golden(INPUT_DATA_TYPE *Data, BIN_DATA_TYPE *Histogram, int data_size, int bin_size) {

	for(int j = 0; j < bin_size; j++) {
			Histogram[j]=0;
	}

	for(int j = 0; j < data_size; j++) {
		Histogram[(unsigned int)Data[j]]++;
	}
}


