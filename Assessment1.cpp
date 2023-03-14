/* 

Rishum - CMP3752M Parallel Programming Assessment 

All the required statistics has been implemented by this program, including the calculation of the Median, 1st Quartile and 3rd Quartile. 
All the statistics are handled and outputted using their exact values (Float). This program as required outputs the Memory Transfer, Kernel 
Execution & Total Program execution times.

To sort the Dataset a Bitonic split sorting algorithm has been implemented and performs well on the shorter dataset. On the longer dataset, 
the sorting process is incredibly slow and will take significant amounts of time to complete dependent on your hardware and selected platform. 
The Mean, Minimum, Maximum and Standard Deviation values are all outputted before the dataset is sorted. The performance values are outputted
at the end of the program and as a result, the sorting process must complete in order to view them.

The program allows you to easily switch datasets and platforms in the initial menus provided and input validation is present to prevent crashes. 
All the Kernels (bar sort) implemented, use Atomic Functions (Add, Max, Min, Xchg) to calculate their relevant values. This program has only been tested on Windows; other
operating systems may have issues.


*/

/* Including the required librarys for this program */
#include <iostream>
#include <vector>
#include "Utils.h"
#include <fstream>
#include <algorithm>

/* Declaring a vector to to store the Temperature dataset*/
vector<float> Temperature;

/* Method to allow the user select a dataset and read it in to the program*/
void Read_Dataset() {


	bool Loop_Selection = false;
	string filename;

	//A simple loop to ensure that User's input is valid.
	do {

		Loop_Selection = false;
		string selection;
		
		cout << "Please enter which Dataset you would like to use:\n\n 1. Short Dataset (18732 Entries) \n 2. Long Dataset (1873106 Entries)\n\n";
		cout << "Option Number: ";
		cin >> selection;
		
		//Clearing the console screen (Windows Only) to improve console readability
		system("cls");


		if (selection == "1") {
			cout << "Loading Small Dataset - This may take a few seconds...\n\n";
			filename = "temp_lincolnshire_short.txt";
		}
		else if (selection == "2") {
			cout << "Loading Large Dataset - This may take a bit...\n\n";
			filename = "temp_lincolnshire.txt";
		}
		else {
			cout << "Invalid Selection Entered!\n\n";
			Loop_Selection = true;

		}
	} while (Loop_Selection == true);

	//Reading in the dataset line by line but ignoring the first five values.
	string line;
	ifstream infile(filename);
	while (getline(infile, line)) 
	{
		float space = 0;
		for (int i = 0; i < 5; i++) {
			space = line.find(' ', space + 1);

		}

		//Inserting it into the vector
		Temperature.push_back(stof(line.substr(space + 1)));

	}

	system("cls");
	cout << "Dataset has finished loading.\n\n";

}

/* A simple Method to check if a string only contains numbers. Used to assist in input validation*/
bool is_number(const std::string& s)
{
	//Loop through each value and check if it is a digit.
	std::string::const_iterator it = s.begin();
	while (it != s.end() && std::isdigit(*it)) ++it;
	return !s.empty() && it == s.end();
}

/* A Method that allows the user to select a platform for OpenCL to run on */
int Platform_Select() {

	int platform_id;
	bool Loop = true;
	
	//Simple do loop for input validation
	do {

		//Listing all the platforms available on the system.
		vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Printing these platforms out.
		int num_platforms = platforms.size();
		for (int i = 0; i < num_platforms; i++)
		{
			cout << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
		}

		//Allowing the user to select a platform

		string User_Selection;
		cout << "\nPlease enter the Platform ID you wish to use from above: ";
		cin >> User_Selection;

		//Some simple input validation to ensure they have only entered a valid platform ID.
		if (is_number(User_Selection)){
			if ((stoi(User_Selection) >= 0) && (stoi(User_Selection) <= (platforms.size() - 1))) {
				platform_id = stoi(User_Selection);
				Loop = false;
			}
			else {
				system("cls");
				cout << "Invalid Platform ID\n\n";
			}
		}
		else {
			system("cls");

			cout << "Invalid Value Entered\n\n";
		}

	} while (Loop);

	system("cls");

	//Returning the selected platform
	return platform_id;

}


/* Main method of this program that contains all the main functionality required. The OpenCL sections
have not been put into their own seperate methods as they will only be invoked once during this program.
*/

int main() {
	
	//Calling the method to read the Dataset into the program.
	Read_Dataset();

	//Calling the method for the user to select a platform and storing this in a varable.
	int platform_id = Platform_Select();
	int device_id = 0;

	//A try catch to catch any potental OpenCL errors.
	try {

		//--- Inital Setup - Creating the Context, Command Queue, Getting & Building Kernals and some inital pre-processing ---// 

		//Creating the context using the platform selected.
		cl::Context context = GetContext(platform_id, device_id);

		//Outputting to the console the current platform and device
		cout << "Currently Platform: " << GetPlatformName(platform_id) << "\nCurrent Device: " << GetDeviceName(platform_id, device_id) << endl;
				

		//Creating the Command Queue with Profiling Enabled
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//Creating the program and building the Kernels.
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		try {
			program.build();
		}
		catch (const cl::Error& err) {
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			throw err;
		}


		//Getting Start time for Profiling
		clock_t Start = clock();

		//Storing Needed Sizes
		size_t InputSize = Temperature.size() * sizeof(float);
		size_t OutputSize = Temperature.size() * sizeof(float);
		size_t LocalSize = 2; //Change Me
		size_t PaddingSize = InputSize % LocalSize;


		//Setting up the events for profiling. 
		cl::Event P_Sum_Input_Buffer_1,P_Sum_Output_Buffer_2,P_Sum_Read_Buffer_3, P_Sum_Kernal;
		cl::Event P_Max_Input_Buffer_1,P_Max_Output_Buffer_2,P_Max_Read_Buffer_3, P_Max_Kernal;
		cl::Event P_Min_Input_Buffer_1, P_Min_Output_Buffer_2, P_Min_Read_Buffer_3, P_Min_Kernal;
		cl::Event P_Var_Input_Buffer_1, P_Var_Output_Buffer_2, P_Var_Read_Buffer_3,  P_Var_Kernal;
		cl::Event P_Sort_Input_Buffer_1, P_Sort_Output_Buffer_2, P_Sort_Read_Buffer_3, P_Sort_Kernal;



		//-- Using OpenCL to calculate the Sum of the dataset and then using it to calculate the datasets mean value. -- // 

		//Creating a result vector at the same size as the data vector.
		vector<float> Sum_Result(Temperature.size());

		//Creating the kernel needed.
		cl::Kernel Kernal_Sum = cl::Kernel(program, "sum");

		//Creating the Buffers which will pass data to and from the Kernal.
		cl::Buffer Sum_Input_Buffer(context, CL_MEM_READ_WRITE, InputSize);
		cl::Buffer Sum_Output_Buffer(context, CL_MEM_READ_WRITE, OutputSize);

		//Passing the temp data into the write buffer.
		queue.enqueueWriteBuffer(Sum_Input_Buffer, CL_TRUE, 0, InputSize, Temperature.data(), NULL, &P_Sum_Input_Buffer_1);

		//Set the output buffer to values of zero
		queue.enqueueFillBuffer(Sum_Output_Buffer, 0, 0, OutputSize, NULL, &P_Sum_Output_Buffer_2);

		//Passing the required args.
		Kernal_Sum.setArg(0, Sum_Input_Buffer);
		Kernal_Sum.setArg(1, Sum_Output_Buffer);
		Kernal_Sum.setArg(2, cl::Local(LocalSize * sizeof(float)));
		
		//Running the Kernal
		queue.enqueueNDRangeKernel(Kernal_Sum, cl::NullRange, cl::NDRange(Temperature.size()), cl::NDRange(LocalSize), NULL, &P_Sum_Kernal);

		//Read the output into the result vector
		queue.enqueueReadBuffer(Sum_Output_Buffer, CL_TRUE, 0, OutputSize, &Sum_Result[0], NULL, &P_Sum_Read_Buffer_3);

		//Getting Sum & Mean and storing them in a varable.
		float Sum = Sum_Result.front();
		float Mean = Sum_Result.front() / Temperature.size();





		//-- Using OpenCL to calculate the Max value of the dataset -- // 

		//Creating a result vector at the same size as the data vector.
		vector<float> Max_Result(Temperature.size());

		//Creating the kernel.
		cl::Kernel Kernal_Max = cl::Kernel(program, "max");

		//Creating the Buffers which will pass data to and from the Kernal.
		cl::Buffer Max_Input_Buffer(context, CL_MEM_READ_WRITE, InputSize);
		cl::Buffer Max_Output_Buffer(context, CL_MEM_READ_WRITE, OutputSize);

		//Passing the temp data into the write buffer.
		queue.enqueueWriteBuffer(Max_Input_Buffer, CL_TRUE, 0, InputSize, Temperature.data() , NULL, &P_Max_Input_Buffer_1);

		//Set the output buffer to values of zero
		queue.enqueueFillBuffer(Max_Output_Buffer, 0, 0, OutputSize, NULL, &P_Max_Output_Buffer_2);

		//Passing the required args.
		Kernal_Max.setArg(0, Max_Input_Buffer);
		Kernal_Max.setArg(1, Max_Output_Buffer);
		Kernal_Max.setArg(2, cl::Local(LocalSize * sizeof(float)));

		//Running the Kernal
		queue.enqueueNDRangeKernel(Kernal_Max, cl::NullRange, cl::NDRange(Temperature.size()), cl::NDRange(LocalSize), NULL, &P_Max_Kernal);
	
		//Read the output into the result vector
		queue.enqueueReadBuffer(Max_Output_Buffer, CL_TRUE, 0, OutputSize, &Max_Result[0], NULL, &P_Max_Read_Buffer_3);
		
		//Storing the Max value in a varable.
		float Max_Value = Max_Result.front();


		//-- Using OpenCL to calculate the Min value of the dataset -- // 
		
		//Creating a result vector at the same size as the data vector.
		vector<float> Min_Result(Temperature.size());

		//Creating the kernel.
		cl::Kernel Kernal_Min = cl::Kernel(program, "min");

		//Creating the Buffers which will pass data to and from the Kernal.
		cl::Buffer Min_Input_Buffer(context, CL_MEM_READ_WRITE, InputSize);
		cl::Buffer Min_Output_Buffer(context, CL_MEM_READ_WRITE, OutputSize);

		//Passing the temp data into the write buffer.
		queue.enqueueWriteBuffer(Min_Input_Buffer, CL_TRUE, 0, InputSize, Temperature.data(), NULL, &P_Min_Input_Buffer_1);

		//Set the output buffer to values of zero
		queue.enqueueFillBuffer(Min_Output_Buffer, 0, 0, OutputSize, NULL, &P_Min_Output_Buffer_2);

		//Passing the required args.
		Kernal_Min.setArg(0, Min_Input_Buffer);
		Kernal_Min.setArg(1, Min_Output_Buffer);
		Kernal_Min.setArg(2, cl::Local(LocalSize * sizeof(float)));

		//Running the Kernal
		queue.enqueueNDRangeKernel(Kernal_Min, cl::NullRange, cl::NDRange(Temperature.size()), cl::NDRange(LocalSize), NULL, &P_Min_Kernal);

		//Read the output into the result vector
		queue.enqueueReadBuffer(Min_Output_Buffer, CL_TRUE, 0, OutputSize, &Min_Result[0], NULL, &P_Min_Read_Buffer_3);

		//Storing the Min value in a varable.
		float Min_Value = Min_Result.front();





		//-- Using OpenCL to calculate the Variance value of the dataset -- // 

		//Creating a result vector at the same size as the data vector.
		vector<float> Variance_Result(Temperature.size());

		//Creating the kernel.
		cl::Kernel Kernal_Variance = cl::Kernel(program, "variance");

		//Creating the Buffers which will pass data to and from the Kernal.
		cl::Buffer Variance_Input_Buffer(context, CL_MEM_READ_WRITE, InputSize);
		cl::Buffer Variance_Output_Buffer(context, CL_MEM_READ_WRITE, OutputSize);

		//Passing the temp data into the write buffer.
		queue.enqueueWriteBuffer(Variance_Input_Buffer, CL_TRUE, 0, InputSize, Temperature.data(), NULL, &P_Var_Input_Buffer_1);

		//Set the output buffer to values of zero
		queue.enqueueFillBuffer(Variance_Output_Buffer, 0, 0, OutputSize, NULL, &P_Var_Output_Buffer_2);

		//Passing the required args.
		Kernal_Variance.setArg(0, Variance_Input_Buffer);
		Kernal_Variance.setArg(1, Variance_Output_Buffer);
		Kernal_Variance.setArg(2, cl::Local(LocalSize * sizeof(float)));
		Kernal_Variance.setArg(3, sizeof(float), &Mean);

		//Running the Kernal
		queue.enqueueNDRangeKernel(Kernal_Variance, cl::NullRange, cl::NDRange(Temperature.size()), cl::NDRange(LocalSize), NULL, &P_Var_Kernal);
		
		//Read the output into the result vector
		queue.enqueueReadBuffer(Variance_Output_Buffer, CL_TRUE, 0, OutputSize, &Variance_Result[0], NULL, &P_Var_Read_Buffer_3);

		//Storing the Std and Variance values in a varable.
		float Variance = Variance_Result.front();
		float Standard_Deviation = sqrt(Variance_Result.front() / Temperature.size());





		//-- Inital printing of the results before sorting the dataset -- // 
		cout << "\n========================= Statistical Results: =========================";

		cout << "\n\nTotal Sum: " << Sum;
		cout << "\nMean Value: " << Mean;
		cout << "\nMaximium Value: " << Max_Value;
		cout << "\nMinimium Value: " << Min_Value;
		cout << "\nStandard Deviation: " << Standard_Deviation;

		cout << "\n\nDataset is now being sorted for the remaining stats. This may take a very long time on the large dataset";





		//-- Using OpenCL to Sort the dataset -- // 

		//Creating a result vector at the same size as the data vector.
		vector<float> Temperature_Sorted(Temperature.size());

		//Creating the kernel.
		cl::Kernel Kernal_Sort = cl::Kernel(program, "sort");

		//Creating the Buffers which will pass data to and from the Kernal.
		cl::Buffer Sorting_Input_Buffer(context, CL_MEM_READ_WRITE, InputSize);
		cl::Buffer Sorting_Output_Buffer(context, CL_MEM_READ_WRITE, InputSize);

		//Passing the temp data into the write buffer.
		queue.enqueueWriteBuffer(Sorting_Input_Buffer, CL_TRUE, 0, InputSize, Temperature.data(), NULL, &P_Sort_Input_Buffer_1);

		//Set the output buffer to values of zero
		queue.enqueueFillBuffer(Sorting_Output_Buffer, 0, 0, OutputSize, NULL, &P_Sort_Output_Buffer_2);
		
		//Passing the required args.
		Kernal_Sort.setArg(0, Sorting_Input_Buffer);
		Kernal_Sort.setArg(1, Sorting_Output_Buffer);
		Kernal_Sort.setArg(2, cl::Local(LocalSize * sizeof(float)));

		//Setting inital Merge to 0.
		int merge = 0;


		Kernal_Sort.setArg(3, merge);


		//Running the Kernal
		queue.enqueueNDRangeKernel(Kernal_Sort, cl::NullRange, cl::NDRange(Temperature.size()), cl::NDRange(LocalSize), NULL, &P_Sort_Kernal);

		//Setting the input arg as the output buffer. 
		Kernal_Sort.setArg(0, Sorting_Output_Buffer); 

		//Looping and rerunning Kernal till the dataset is sorted.
		bool Sorted = false;
		while (Sorted == false) 
		{
			//Flipping merge value
			Kernal_Sort.setArg(3, merge); 

			if (merge == 1){
				merge = 0;
			} else if (merge == 0) {
				merge = 1;
			}

			//Running the Kernal
			queue.enqueueNDRangeKernel(Kernal_Sort, cl::NullRange, cl::NDRange(Temperature.size()), cl::NDRange(LocalSize), NULL, &P_Sort_Kernal);
			
			//Read the output into the result vector
			queue.enqueueReadBuffer(Sorting_Output_Buffer, CL_TRUE, 0, OutputSize, &Temperature_Sorted[0],NULL, &P_Sort_Read_Buffer_3);
																
			//Check if data is sorted
			if (is_sorted(Temperature_Sorted.begin(), Temperature_Sorted.end()))
			{
				Sorted = true;
			}


		}

		//Using sorted dataset to get the median and quartiles
		float Median = Temperature_Sorted[0.5 * Temperature_Sorted.size()];
		float Quartile_1 = Temperature_Sorted[0.25 * Temperature_Sorted.size()];
		float Quartile_3 = Temperature_Sorted[0.75 * Temperature_Sorted.size()];


		//Outputting these values
		cout << "\n\nMedian: " << Median;
		cout << "\n1st Quartile: " << Quartile_1;
		cout << "\n3rd Quartile: " << Quartile_3;



		//-- Performance Result Output Section -- // 

		cout << "\n\n========================= Performance Results: =========================";
		
		//Calcuating and Outputting the Sum Kernal's performance times

		auto R_Sum_Input_Buffer_1 = P_Sum_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sum_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Sum_Output_Buffer_2 = P_Sum_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sum_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Sum_Read_Buffer_3 = P_Sum_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sum_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Sum_Kernal = P_Sum_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sum_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		
		cout << "\n\nSum Kernal Memory Transfer (Input Buffer): " << R_Sum_Input_Buffer_1 << "ns";
		cout << "\nSum Kernal Memory Transfer (Output Buffer): " << R_Sum_Output_Buffer_2 << "ns";
		cout << "\nSum Kernal Memory Transfer (Read Buffer): " << R_Sum_Read_Buffer_3 << "ns";
		cout << "\nSum Kernal Total Execution Time: " << R_Sum_Kernal << "ns";

		//Calcuating and Outputting the Max Kernal's performance times

		auto R_Max_Input_Buffer_1 = P_Max_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Max_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Max_Output_Buffer_2 = P_Max_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Max_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Max_Read_Buffer_3 = P_Max_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Max_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Max_Kernal = P_Max_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Max_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		
		cout << "\n\nMax Kernal Memory Transfer (Input Buffer): " << R_Max_Input_Buffer_1 << "ns";
		cout << "\nMax Kernal Memory Transfer (Output Buffer): " << R_Max_Output_Buffer_2 << "ns";
		cout << "\nMax Kernal Memory Transfer (Read Buffer): " << R_Max_Read_Buffer_3 << "ns";
		cout << "\nMax Kernal Total Execution Time: " << R_Max_Kernal << "ns";

		//Calcuating and Outputting the Min Kernal's performance times

		auto R_Min_Input_Buffer_1 = P_Min_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Min_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Min_Output_Buffer_2 = P_Min_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Min_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Min_Read_Buffer_3 = P_Min_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Min_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Min_Kernal = P_Min_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Min_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();

		cout << "\n\nMin Kernal Memory  Transfer (Input Buffer): " << R_Min_Input_Buffer_1 << "ns";
		cout << "\nMin Kernal Memory Transfer (Output Buffer): " << R_Min_Output_Buffer_2 << "ns";
		cout << "\nMin Kernal Memory Transfer (Read Buffer): " << R_Min_Read_Buffer_3 << "ns";
		cout << "\nMin Kernal Total Execution Time: " << R_Min_Kernal << "ns";

		//Calcuating and Outputting the Variance Kernal's performance times

		auto R_Var_Input_Buffer_1 = P_Var_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Var_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Var_Output_Buffer_2 = P_Var_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Var_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Var_Read_Buffer_3 = P_Var_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Var_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Var_Kernal = P_Var_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Var_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();

		cout << "\n\nVar Kernal Memory Transfer (Input Buffer): " << R_Var_Input_Buffer_1 << "ns";
		cout << "\nVar Kernal Memory Transfer (Output Buffer): " << R_Var_Output_Buffer_2 << "ns";
		cout << "\nVar Kernal Memory Transfer (Read Buffer): " << R_Var_Read_Buffer_3 << "ns";
		cout << "\nVar Kernal Total Execution Time: " << R_Var_Kernal << "ns";

		//Calcuating and Outputting the Sorting Kernal's performance times

		auto R_Sort_Input_Buffer_1 = P_Sort_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sort_Input_Buffer_1.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Sort_Output_Buffer_2 = P_Sort_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sort_Output_Buffer_2.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Sort_Read_Buffer_3 = P_Sort_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sort_Read_Buffer_3.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		auto R_Sort_Kernal = P_Sort_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_END>() - P_Sort_Kernal.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();

		cout << "\n\nSort Kernal Memory Transfer (Input Buffer): " << R_Sort_Input_Buffer_1 << "ns";
		cout << "\nSort Kernal Memory Transfer (Output Buffer): " << R_Sort_Output_Buffer_2 << "ns";
		cout << "\nSort Kernal Memory Transfer (Read Buffer): " << R_Sort_Read_Buffer_3 << "ns";
		cout << "\nSort Kernal Total Execution Time: " << R_Sort_Kernal << "ns";

		//Calcuating and Outputting the Total Program's performance times

		cout << "\n\nTotal Program Execution Time: " << (((float) clock() - Start) / CLOCKS_PER_SEC) << "s";

		//Pausing the program once it has finished.
		cin.ignore();
		cin.get();
	

	}
	//Catching the error and outputting it to console.
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}	
}