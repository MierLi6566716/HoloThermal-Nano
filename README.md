# HoloThermal-Nano
A thermal-aware thread scheduling and edge-cloud partitioning framework for   holographic AR workloads on the NVIDIA Jetson Nano. Reduces peak CPU temperatures by   3.0°C and prevents thermal throttling using predictive thermal-slope scheduling and   ResNet-18 mid-layer offloading.

## Key Contributions
*   **Predictive Headroom Scheduler:** A custom thread-to-core scheduler that
      monitors both instantaneous temperature and the rate of thermal rise (thermal
      slope) to proactively manage core concurrency.
*   **Loople Trajectory Scheduler:** A learning-based scheduler that navigates a
      multi-dimensional parameter space (temperature thresholds and dwell times) to
      optimize the balance between latency and heat.
*   **Edge-Cloud Partitioning:** A mid-layer partitioning strategy for the
      ResNet-18 + DPRC pipeline that offloads compute-intensive holographic generation
      to a cloud co-processor.
