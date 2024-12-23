#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace param {
    const int n_steps = 200000;
    const double dt = 60;
    const double eps = 1e-3;
    const double G = 6.674e-11;
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;

    __device__ __host__ double gravity_device_mass(double m0, double t) {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000));
    }

    __device__ __host__ double get_missile_cost(double t) {
        return 1e5 + 1e3 * t;
    }
} // namespace param

// Adjust as needed
static constexpr int BLOCK_SIZE = 256;

// GPU kernel for computing accelerations with tiling and shared memory
__global__ void compute_accelerations(int n, const double* __restrict__ qx, const double* __restrict__ qy, 
                                      const double* __restrict__ qz, double* __restrict__ ax, double* __restrict__ ay, 
                                      double* __restrict__ az, const double* __restrict__ m, 
                                      const char* __restrict__ type_flags, double t) 
{
    extern __shared__ double shmem[]; 
    // Layout of shared memory:
    // First BLOCK_SIZE doubles: sh_qx
    // Next BLOCK_SIZE doubles:  sh_qy
    // Next BLOCK_SIZE doubles:  sh_qz
    // Next BLOCK_SIZE doubles:  sh_m
    // Next BLOCK_SIZE chars:    sh_type
    double* sh_qx = shmem;
    double* sh_qy = shmem + BLOCK_SIZE;
    double* sh_qz = shmem + 2 * BLOCK_SIZE;
    double* sh_m  = shmem + 3 * BLOCK_SIZE;
    char*   sh_type = (char*)(sh_m + BLOCK_SIZE);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double xi = qx[i];
    double yi = qy[i];
    double zi = qz[i];

    double axi = 0.0;
    double ayi = 0.0;
    double azi = 0.0;

    int tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < tiles; tile++) {
        int idx = tile * BLOCK_SIZE + threadIdx.x;
        if (idx < n) {
            sh_qx[threadIdx.x] = qx[idx];
            sh_qy[threadIdx.x] = qy[idx];
            sh_qz[threadIdx.x] = qz[idx];
            sh_m[threadIdx.x]  = m[idx];
            sh_type[threadIdx.x] = type_flags[idx];
        } else {
            // For out-of-range threads, set dummy values
            sh_qx[threadIdx.x] = 0.0;
            sh_qy[threadIdx.x] = 0.0;
            sh_qz[threadIdx.x] = 0.0;
            sh_m[threadIdx.x]  = 0.0;
            sh_type[threadIdx.x] = 0;
        }
        __syncthreads();

        // Compute partial acceleration contribution from this tile
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int body_j = tile * BLOCK_SIZE + j;
            if (body_j == i || body_j >= n) continue;

            double mj = sh_m[j];
            if (sh_type[j] == 1) { // device type
                mj = param::gravity_device_mass(mj, t);
            }

            double dx = sh_qx[j] - xi;
            double dy = sh_qy[j] - yi;
            double dz = sh_qz[j] - zi;

            double dist2 = dx * dx + dy * dy + dz * dz + param::eps * param::eps;
            double invDist = rsqrt(dist2); // reciprocal sqrt
            double invDist3 = invDist * invDist * invDist;

            double fac = param::G * mj * invDist3;
            axi += fac * dx;
            ayi += fac * dy;
            azi += fac * dz;
        }

        __syncthreads();
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

// GPU kernel for updating positions and velocities
__global__ void update_positions_velocities(int n, double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
                                            double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
                                            const double* __restrict__ ax, const double* __restrict__ ay, const double* __restrict__ az, double dt) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double vxi = vx[i] + ax[i] * dt;
    double vyi = vy[i] + ay[i] * dt;
    double vzi = vz[i] + az[i] * dt;

    qx[i] += vxi * dt;
    qy[i] += vyi * dt;
    qz[i] += vzi * dt;

    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
}

// GPU kernel for checking collision
__global__ void check_collision_kernel(const double* __restrict__ qx, const double* __restrict__ qy, 
                                       const double* __restrict__ qz, int planet, int asteroid, bool* collision) 
{
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    *collision = (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius);
}

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

class GPUSimulation {
private:
    int n;
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
    double *d_ax, *d_ay, *d_az;
    char* d_type_flags;
    bool* d_collision;

public:
    GPUSimulation(int n, const std::vector<double>& qx, const std::vector<double>& qy,
        const std::vector<double>& qz, const std::vector<double>& vx,
        const std::vector<double>& vy, const std::vector<double>& vz,
        const std::vector<double>& m, const std::vector<std::string>& type) : n(n) {
        
        hipMalloc(&d_qx, n * sizeof(double));
        hipMalloc(&d_qy, n * sizeof(double));
        hipMalloc(&d_qz, n * sizeof(double));
        hipMalloc(&d_vx, n * sizeof(double));
        hipMalloc(&d_vy, n * sizeof(double));
        hipMalloc(&d_vz, n * sizeof(double));
        hipMalloc(&d_m, n * sizeof(double));
        hipMalloc(&d_ax, n * sizeof(double));
        hipMalloc(&d_ay, n * sizeof(double));
        hipMalloc(&d_az, n * sizeof(double));
        hipMalloc(&d_type_flags, n * sizeof(char));
        hipMalloc(&d_collision, sizeof(bool));

        std::vector<char> type_flags(n);
        for (int i = 0; i < n; i++) {
            type_flags[i] = (type[i] == "device") ? 1 : 0;
        }

        hipMemcpy(d_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_m, m.data(), n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_type_flags, type_flags.data(), n * sizeof(char), hipMemcpyHostToDevice);
    }

    ~GPUSimulation() {
        hipFree(d_qx);
        hipFree(d_qy);
        hipFree(d_qz);
        hipFree(d_vx);
        hipFree(d_vy);
        hipFree(d_vz);
        hipFree(d_m);
        hipFree(d_ax);
        hipFree(d_ay);
        hipFree(d_az);
        hipFree(d_type_flags);
        hipFree(d_collision);
    }

    void run_step(int step) {
        int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double t = step * param::dt;

        // Compute the size of shared memory needed
        // 4 arrays of doubles (each BLOCK_SIZE) + 1 array of chars (BLOCK_SIZE)
        size_t shared_mem_size = (4 * BLOCK_SIZE * sizeof(double)) + (BLOCK_SIZE * sizeof(char));

        // Compute accelerations
        hipLaunchKernelGGL(compute_accelerations, dim3(num_blocks), dim3(BLOCK_SIZE), shared_mem_size, 0,
            n, d_qx, d_qy, d_qz, d_ax, d_ay, d_az, d_m, d_type_flags, t);

        // Update positions and velocities
        hipLaunchKernelGGL(update_positions_velocities, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0, 
            n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, param::dt);
    }

    bool check_collision(int planet, int asteroid) {
        hipLaunchKernelGGL(check_collision_kernel, dim3(1), dim3(1), 0, 0,
            d_qx, d_qy, d_qz, planet, asteroid, d_collision);
        bool collision;
        hipMemcpy(&collision, d_collision, sizeof(bool), hipMemcpyDeviceToHost);
        return collision;
    }

    double get_min_distance(int planet, int asteroid) {
        std::vector<double> h_qx(n), h_qy(n), h_qz(n);
        hipMemcpy(h_qx.data(), d_qx, n * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(h_qy.data(), d_qy, n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(h_qz.data(), d_qz, n * sizeof(double), hipMemcpyHostToDevice);

        double dx = h_qx[planet] - h_qx[asteroid];
        double dy = h_qy[planet] - h_qy[asteroid];
        double dz = h_qz[planet] - h_qz[asteroid];
        return sqrt(dx * dx + dy * dy + dz * dz);
    }

    void reset_device(const std::vector<double>& m) {
        hipMemcpy(d_m, m.data(), n * sizeof(double), hipMemcpyHostToDevice);
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    // Problem 1: Find minimum distance
    double min_dist = std::numeric_limits<double>::infinity();
    std::vector<double> m_no_devices = m;
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m_no_devices[i] = 0;
        }
    }

    {
        GPUSimulation sim1(n, qx, qy, qz, vx, vy, vz, m_no_devices, type);
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                sim1.run_step(step);
            }
            double dist = sim1.get_min_distance(planet, asteroid);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }

    // Problem 2: Find collision time
    int hit_time_step = -2;
    {
        GPUSimulation sim2(n, qx, qy, qz, vx, vy, vz, m, type);
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                sim2.run_step(step);
            }
            if (sim2.check_collision(planet, asteroid)) {
                hit_time_step = step;
                break;
            }
        }
    }

    // Problem 3: Find optimal device to destroy
    int gravity_device_id = -1;
    double missile_cost = 0;
    bool found_solution = false;

    for (int device_id = 0; device_id < n; device_id++) {
        if (type[device_id] != "device") continue;

        std::vector<double> temp_m = m;
        GPUSimulation sim3(n, qx, qy, qz, vx, vy, vz, temp_m, type);

        bool will_collide = false;
        int hit_step = -1;

        for (int step = 1; step <= param::n_steps; step++) {
            if (hit_step == -1) {
                double device_dist = sim3.get_min_distance(planet, device_id);
                double missile_dist = step * param::dt * param::missile_speed;

                if (missile_dist > device_dist) {
                    hit_step = step;
                    temp_m[device_id] = 0;
                    sim3.reset_device(temp_m);
                }
            }

            sim3.run_step(step);

            if (sim3.check_collision(planet, asteroid)) {
                will_collide = true;
                break;
            }
        }

        if (!will_collide && hit_step != -1) {
            double time = hit_step * param::dt;
            double cost = param::get_missile_cost(time);

            if (!found_solution || cost < missile_cost) {
                found_solution = true;
                gravity_device_id = device_id;
                missile_cost = cost;
            }
        }
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    return 0;
}
