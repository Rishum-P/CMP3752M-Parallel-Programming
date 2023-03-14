//Kernel to find the Minimum value using Atomic Min
__kernel void min(__global const float* input, __global float* output, __local float* local_cache) {

	//Get required item infos
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	
	//Copy input into local memory
	local_cache[local_id] = input[id];

	//Pause to allow threads to catch up
	barrier(CLK_LOCAL_MEM_FENCE); 

	//Calculate Min Value
	for (int i = local_size / 2; i > 0; i /= 2) {
		if (local_id < i) {
			local_cache[local_id] = min(local_cache[local_id + i], local_cache[local_id]);
		}
		barrier(CLK_LOCAL_MEM_FENCE); 
	}
	if (!local_id) {
		float new_val = local_cache[local_id];
		while (new_val != 0.0) {
			float old_val = atomic_xchg(&output[0], 0.0); 

			new_val = atomic_xchg(&output[0], min(old_val, new_val));
		}
	}
}

//Kernel to find the Minimum value using Atomic Max
__kernel void max(__global const float* input, __global float* output, __local float* local_cache) {

	//Get required item infos
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy input into local memory
	local_cache[local_id] = input[id];

	//Pause to allow threads to catch up
	barrier(CLK_LOCAL_MEM_FENCE); 

	//Calculate Max Value. Same as Min value but with max instead of min
	for (int i = local_size / 2; i > 0; i /= 2) {
		if (local_id < i) {
			local_cache[local_id] = max(local_cache[local_id + i], local_cache[local_id]);
		}
		barrier(CLK_LOCAL_MEM_FENCE); 
	}
	if (!local_id) {
		float new_val = local_cache[local_id];
		while (new_val != 0.0) {
			float old_val = atomic_xchg(&output[0], 0.0); 
			new_val = atomic_xchg(&output[0], max(old_val, new_val)); 
		}
	}
}

//Calculate Sum Value using Atomic add
__kernel void sum(__global const float* input, __global float* output, __local float* local_cache) {
	
	//Get required item infos
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	
	//Copy input into local memory
	local_cache[local_id] = input[id];

	//Pause to allow threads to catch up
	barrier(CLK_LOCAL_MEM_FENCE); 

	//Calculate Sum
	for (int i = local_size/2; i > 0; i /= 2) {
		if(local_id < i){
			local_cache[local_id] += local_cache[local_id + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE); 
	}
	if (!local_id) {
		float add_val = local_cache[local_id];
		while (add_val != 0.0) {

			float old_val = atomic_xchg(&output[0], 0.0);
			add_val = atomic_xchg(&output[0], old_val + add_val); 
		}
	}
}

//Calculate variance Value using Atomic xchg
__kernel void variance(__global const float* input, __global float* output, __local float* local_cache, const float mean) {
	
	//Get required item infos
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy input into local memory
	local_cache[local_id] = pow((mean-input[id]),2); 
	
	//Pause to allow threads to catch up
	barrier(CLK_LOCAL_MEM_FENCE); 

	//Calculate variance
	for (int i = local_size / 2; i > 0; i /= 2) {
		if (local_id < i) {
			local_cache[local_id] += local_cache[local_id + i]; 
		}
		barrier(CLK_LOCAL_MEM_FENCE); 
	}
	if (!local_id) {
		float add_val = local_cache[local_id];
		while (add_val != 0.0) {
			float old_val = atomic_xchg(&output[0], 0.0); 
			add_val = atomic_xchg(&output[0], old_val + add_val); 
		}
	}
}

//Sort an array using Bitonic sort 
__kernel void sort(__global const float* input, __global float* output, __local float* local_cache, int merge)
{
	//Get required item infos
    int id = get_global_id(0);
    int local_id = get_local_id(0);    
    int local_size = get_local_size(0);
	
    int offset = id + ((local_size/2) * merge); 

	//Copy input into local memory
    local_cache[local_id] = input[offset];

	//Pause to allow threads to catch up
    barrier(CLK_LOCAL_MEM_FENCE); 

	//Sort Data
    for (int i = 1; i < local_size; i <<= 1) 
    {
        bool direction = ((local_id & (i <<1)) != 0);

        for (int inc = i; inc > 0; inc >>= 1)	
        {										
            int j = local_id ^ inc;
            float i_data = local_cache[local_id];
            float j_data = local_cache[j];			

            bool smaller = (j_data < i_data) || ( j_data == i_data && j < local_id);
            bool swap = smaller ^ (j < local_id) ^ direction; 

            barrier(CLK_LOCAL_MEM_FENCE);
            local_cache[local_id] = (swap) ? j_data : i_data; 
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    output[offset] = local_cache[local_id];    

}