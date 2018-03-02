/* File: histogram.cl
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
* File name : histogram.cl
* author    : Mohammad hosseinabady mohammad@hosseinabady.com
* date      : 1 October 2017
* blog: https://highlevel-synthesis.com/
*/
#include "../host/histogram.h"


//pipe  INPUT_DATA_TYPE pdata __attribute__((xcl_reqd_pipe_depth(PIPE_DEPTH)));
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel  INPUT_DATA_TYPE pdata;



__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
read_data_kernel(__global INPUT_DATA_TYPE* vectorData, int data_length) {


	data_length = DATA_LENGTH;
	//__attribute__((xcl_pipeline_loop))
	#pragma ii 1
	for (int i = 0; i < DATA_LENGTH; i++) {
		//write_pipe_block(pdata, &vectorData[i]);
		write_channel_intel(pdata, &vectorData[i]);
	}

}




__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
compute_data_histogram_kernel(int data_length, int bin_size, __global BIN_DATA_TYPE *hist) {

	data_length = DATA_LENGTH;
	bin_size = BIN_SIZE;

	local BIN_DATA_TYPE  hist_local[BIN_SIZE];


	for (int i = 0; i < BIN_SIZE; i++) {
		hist_local[i] = 0;
	}


	INPUT_DATA_TYPE d_1;
	unsigned int        index_1;
	BIN_DATA_TYPE  hist_1;

	//__attribute__((xcl_pipeline_loop))
	#pragma ii 1
	for (int i = 0; i < data_length; i+=1) {

		//read_pipe_block(pdata, &d_1);
		d_1 = read_channel_intel(pdata);

		index_1 = (unsigned int)d_1;

		hist_local[index_1]++;
	}

	async_work_group_copy(hist, hist_local, BIN_SIZE, 0);

}



