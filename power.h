/*
 * power.h
 *
 *  Created on: Mar 10, 2016
 *      Author: jmilet
 */

#ifndef POWER_H_
#define POWER_H_

#include <pthread.h>
#include <iostream>
#include <vector>

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>

#include <nvml.h>

using namespace std;

class GPUPower {

private:

	int pdevice ;

	volatile bool stop ;
	volatile bool start ;

	int microseconds ;
	int idx;
	pthread_t my_thread ;

	vector <int> gpuSM;
	vector <int> gpuMem;
	vector <int> gpuPower;
	vector <long> gpuTime;


public:

	// Constructor

	GPUPower( int d, int idx ) : pdevice(d), idx(idx), start(false), stop(false)  {

		microseconds = 100 ;

	} ;

	/**
	 * Set start
	 */

	void setStart( bool s )  {

		start = s;
	}


	/**
	 * set the stop flag
	 */

	void setStop( bool s)  {

		stop = s;
	}

	/**
	 * -----------------------------------------------------
	 * Query device properties
	 * -----------------------------------------------------
	 */

	int queryDeviceInfo ( int iteration, int ndevice) {

		nvmlReturn_t result ;
		nvmlReturn_t r;
		nvmlUtilization_t utilization;

		unsigned int device_count , mWpower  ;

		// First initialize NVML library
		result = nvmlInit();

		if (NVML_SUCCESS != result)    {

			printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
			return 0 ;
		}

		result = nvmlDeviceGetCount(&device_count);

		//printf("Found %d device%s\n\n", device_count, device_count != 1 ? "s" : "");

		nvmlDevice_t device;

		char name[NVML_DEVICE_NAME_BUFFER_SIZE];

		nvmlPciInfo_t pci;

		result = nvmlDeviceGetHandleByIndex(ndevice, &device);

		if (NVML_SUCCESS != result)
		{
			printf("Failed to get handle for device %i: %s\n", ndevice, nvmlErrorString(result));
			return 0 ;
		}

		result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);

		if (NVML_SUCCESS != result)
		{
			printf("Failed to get name of device %i: %s\n", ndevice, nvmlErrorString(result));
			return 0 ;
		}

		// pci.busId is very useful to know which device physically you're talking to
		// Using PCI identifier you can also match nvmlDevice handle to CUDA device.

		result = nvmlDeviceGetPciInfo(device, &pci);

		if (NVML_SUCCESS != result)	{

			printf("Failed to get pci info for device %i: %s\n", ndevice, nvmlErrorString(result));
			return 0 ;
		}

		if ((r = nvmlDeviceGetUtilizationRates(device, &utilization)) != NVML_SUCCESS) {
			printf("%s\n", nvmlErrorString(r));
		}

		if ((r = nvmlDeviceGetPowerUsage(device, &mWpower)) != NVML_SUCCESS) {
			printf("%s\n", nvmlErrorString(r));
		}

		//  printf("%d. %s [%s]\n", ndevice, name, pci.busId);
		//  printf("\t\tUtilization: %d ", iteration );
		//	printf("%d%% GPU, %d%% MEM ", utilization.gpu, utilization.memory);
		//  printf(" Power usage: ");
		//  printf(" %dW \n", k/1000);

		// --------------------------------------------------------
		// Get time of the day
		// --------------------------------------------------------

		struct timeval timeValue;
		gettimeofday( &timeValue, NULL ) ;

		// Writing values to vectors
		gpuTime.push_back(timeValue.tv_sec*1000000 + timeValue.tv_usec ) ;
		gpuSM.push_back(utilization.gpu);
		gpuMem.push_back(utilization.memory);
		gpuPower.push_back(mWpower);

		result = nvmlShutdown();

		if (NVML_SUCCESS != result)  {
			printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
		}

		return mWpower ;

	}

	/**
	 * -----------------------------------------------------------------------
	 *  Write to file
	 * -----------------------------------------------------------------------
	 */


	void writeFile()   {
		string str;
		std::ostringstream ss;
		ss << idx;
		str = "powerprofile" + ss.str() + ".txt";
		std::ofstream f(str.c_str());

		int index = 0 ;

		for(vector<int>::const_iterator i = gpuPower.begin(); i != gpuPower.end(); ++i) {

			f << "Time | " << gpuTime.at(index) << " | Power | " << *i << " | Mem |  " << gpuMem.at(index)
										<< " | GPU |  " << gpuSM.at(index) << " | Delta T | " <<  microseconds << " | \n";

			index ++ ;

		}

	}

	/**
	 * -----------------------------------------------------------------------
	 *  Power thread
	 * -----------------------------------------------------------------------
	 */


	void *worker_thread() {

		int index = 0 ;

		unsigned int mWpower  ;

		int iterations = 0 ;

		while ( !stop )  {

			if (start) {

				mWpower = queryDeviceInfo( index++, pdevice ) ;

				iterations++ ;
			}

			usleep(microseconds);

		}

		// Waiting until power goes down

		while ( iterations > 0 )  {

			mWpower = queryDeviceInfo( index++, pdevice ) ;

			usleep(microseconds);

			iterations -- ;

		}

		// Write the power log file

		writeFile() ;

		pthread_exit(NULL);

	}


	static void *helper(void *context )    {

		return ((GPUPower *)context)->worker_thread( );

	}

	/**
	 * -----------------------------------------------------------------------
	 *  Start the power thread
	 * -----------------------------------------------------------------------
	 */

	void startPowerThread( )   {

		int ret =  pthread_create(
				&my_thread,
				NULL,
				&GPUPower::helper,
				this );

		if(ret != 0) {

			printf(" Error in thread creation ...\n");

		}


	}

	/**
	 * -----------------------------------------------------------------------
	 *  Stop the power thread
	 * -----------------------------------------------------------------------
	 */

	void stopPowerThread(  )   {

		void *ret_join;

		int ret = pthread_join( my_thread, &ret_join);

		if(ret != 0) {

			printf(" Pthread_join failed..\n");

			return  ;

		}


	}



};


#endif /* POWER_H_ */
