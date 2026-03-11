# src/cuda_simulator.py
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32
import pandas as pd
import os

# 1. Define CUDA Kernel for Parallel Simulation
@cuda.jit
def simulate_gpu_telemetry(voltage_arr, temp_arr, memory_arr, status_arr, rng_states, n_events):
    idx = cuda.grid(1)
    if idx < n_events:
        # Generate random noise directly on the GPU
        v_noise = xoroshiro128p_normal_float32(rng_states, idx) * 0.05
        t_noise = xoroshiro128p_normal_float32(rng_states, idx) * 3.0
        m_noise = xoroshiro128p_normal_float32(rng_states, idx) * 5.0
        chance = xoroshiro128p_uniform_float32(rng_states, idx)

        # Base physical state (Healthy)
        base_voltage = 1.0 + v_noise
        base_temp = 45.0 + t_noise
        base_memory = 40.0 + m_noise

        # Inject Complex Multi-Class Failures (15% total failure rate)
        if chance < 0.05:
            # Type 1: Power Delivery Instability (High voltage, temp follows slightly)
            base_voltage += 0.35 + (v_noise * 2)
            base_temp += 12.0
            status_arr[idx] = 1 
            
        elif chance < 0.10:
            # Type 2: Thermal Throttling/Crash (Massive temp spike, voltage drops as it throttles)
            base_temp += 38.0 + (t_noise * 4)
            base_voltage -= 0.15
            status_arr[idx] = 2 
            
        elif chance < 0.15:
            # Type 3: Memory Leak / VRAM Pressure (Memory maxes out, slight temp rise)
            base_memory = 96.0 + (m_noise * 3)
            if base_memory > 100.0: 
                base_memory = 100.0
            base_temp += 8.0
            status_arr[idx] = 3 
            
        else:
            # Type 0: Stable Operation (85% of the time)
            status_arr[idx] = 0 
            
        # Write to global memory arrays
        voltage_arr[idx] = base_voltage
        temp_arr[idx] = base_temp
        memory_arr[idx] = base_memory

# 2. Main Execution
def run_simulation(n_events=1000000):
    print(f"Starting CUDA Simulation for {n_events} events...")
    
    # Allocate Host Memory
    h_voltage = np.zeros(n_events, dtype=np.float32)
    h_temp = np.zeros(n_events, dtype=np.float32)
    h_memory = np.zeros(n_events, dtype=np.float32)
    h_status = np.zeros(n_events, dtype=np.int32)
    
    # Allocate Device Memory
    d_voltage = cuda.to_device(h_voltage)
    d_temp = cuda.to_device(h_temp)
    d_memory = cuda.to_device(h_memory)
    d_status = cuda.to_device(h_status)
    
    # Configure Threads (A100 has many cores)
    threads_per_block = 256
    blocks_per_grid = (n_events + threads_per_block - 1) // threads_per_block
    
    # Create RNG states for the GPU
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=42)
    
    # Launch Kernel
    simulate_gpu_telemetry[blocks_per_grid, threads_per_block](
        d_voltage, d_temp, d_memory, d_status, rng_states, n_events
    )
    
    # Wait for GPU to finish
    cuda.synchronize()
    
    # Copy back to Host
    h_voltage = d_voltage.copy_to_host()
    h_temp = d_temp.copy_to_host()
    h_memory = d_memory.copy_to_host()
    h_status = d_status.copy_to_host()
    
    # Save to Scratch (Avoid filling /home)
    df = pd.DataFrame({
        'voltage': h_voltage,
        'temperature': h_temp,
        'memory_util': h_memory,
        'failure_label': h_status
    })
    
    # Use /scratch or your specific data folder
    output_path = os.path.expanduser("~/AMDProjects/gpu-failure-platform/data/telemetry_1m.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    run_simulation()

