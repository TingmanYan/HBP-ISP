/*
 *  Hierarchical Belief Propagation on Image Segmentation Pyramid.
 *  Author: Tingman Yan (tmyann@outlook.com)
 */
#include "hbp_isp.hpp"
#include <cfloat>

// compute the KL divergence for all boundary connection
// sum over all possiable labels
__global__ void
compute_KL_k(const float *const im_cost_d, const int2 *const bd_d, float *const KL_d, const int num_clus, const int m_labels, const int num_bd)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_bd)
        return;

    int s = bd_d[id].x;
    int t = bd_d[id].y;

    float kl = 0;
    for (int d = 0; d < m_labels; ++d)
    {
        float p_s = im_cost_d[s + d * num_clus];
        float p_t = im_cost_d[t + d * num_clus];
        kl += (p_t - p_s) * (expf(-p_s) - expf(-p_t));
        // this is the same as first transform to exp(-cost(d)),
        // then compute KL divergence
    }
    KL_d[id] = kl;
}

// for a vector v, set v[i] = i
__global__ void
set_to_identity(int *, const int);

// get the image-wise cluster (superpixel) label
__global__ void
update_image_label(const int *const clus_d, const int *const im_clus_d,
                   int *const im_clus_h_d, const int im_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= im_size)
        return;

    int cc = im_clus_d[id];
    im_clus_h_d[id] = clus_d[cc];
}

// for a vertor v, set v[i] = T
template <typename T>
__global__ void
set_to_zero(T *const, const int, const T);

// copy the best plane label in the previous level to the prior plane in
// the current level
__global__ void
copy_prior_plane(const int *const clus_d, const float4 *const plane_h_d,
                 float4 *const plane_d, const int num_clus,
                 const int num_clus_h, const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    int cc = clus_d[id];
    for (int p = 0; p < P; ++p)
    {
        float4 prior_best_plane = plane_h_d[cc + p * num_clus_h];
        plane_d[id + p * num_clus] = prior_best_plane;
        plane_d[id + (p + P) * num_clus] = prior_best_plane;
        plane_d[id + (p + P * 2) * num_clus] = prior_best_plane;
    }
}

__global__ void
sum_to_mean(float *, const int, const int);

__global__ void
compute_dist_color_pos_k(const float *const mean_d, const int2 *const bd_d,
                         float *const dc_d, const int num_bd,
                         const int num_clus, const int channels)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= num_bd || y >= channels)
        return;

    int2 bd = bd_d[x];
    int s = bd.x, t = bd.y;
    int index_s = y * num_clus + s, index_t = y * num_clus + t;
    float mean_s = mean_d[index_s], mean_t = mean_d[index_t];
    if (y < 3)
    { // for BGR
        float dist = abs(mean_s - mean_t);
        atomicAdd(&dc_d[x], dist);
    }
    else
    { // for pos
        float dist = (mean_s - mean_t) * (mean_s - mean_t);
        atomicAdd(&dc_d[num_bd + x], dist);
    }
}

__global__ void
dist_trans(float *const dc_d, const float gamma, const int num_bd,
           const int channels)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= num_bd || y >= channels)
        return;

    int id = y * num_bd + x;
    float dist = dc_d[id];
    if (y == 0)
    {
        dist = exp(-dist / gamma);
    }
    else
    {
        dist = sqrtf(dist);
    }
    dc_d[id] = dist;
}

__global__ void
sum_reduce_num_clus(const int *const clus_sort_d, int *const clus_num_pos_d,
                    const int num_clus)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    int c = clus_sort_d[id];
    atomicAdd(&clus_num_pos_d[c], 1);
}

__global__ void
sum_reduce_num_nbs(const int2 *const bd_d, int *const nb_num_pos_d,
                   const int num_bd)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_bd)
        return;

    int s = bd_d[id].x;
    atomicAdd(&nb_num_pos_d[s], 1);
}

// for a bd[id] = s -> t, find the corresponding index reverse_id_d[id],
// such that bd[reverse_id_d[id]] = t -> s
__global__ void
reverse_bd_id_k(const int2 *const bd_d, const int *const pos_scan_d,
                int *const reverse_id_d, const int num_bd)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= num_bd)
        return;

    int2 bd = bd_d[id];
    int s = bd.x, t = bd.y; // s -> t
    // find the id of the corresponding t -> s
    // - since the neighbors of a cluster is small, the speed is fast
    for (int r_id = pos_scan_d[t]; r_id < pos_scan_d[t + 1]; ++r_id)
    {
        if (bd_d[r_id].y == s)
        {
            reverse_id_d[id] = r_id;
            break;
        }
    }
}

__global__ void
read_plane_cost(const float4 *const plane_d, const float4 *const im_D4_d,
                const float *const im_cost_d, float *const im_plane_cost_d,
                const bool use_cost_vol, const float tau, const int im_size,
                const int width, const int m_labels, const float inlier_thrsh,
                const float lambda, const int num_clus, const int s_c,
                const int P)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= im_size)
        return;

    if (!use_cost_vol)
    {
        int cc = im_D4_d[id].x;
        float D = im_D4_d[id].y;
        int im_id = im_D4_d[id].z;
        int x = im_id % width;
        int y = im_id / width;
        for (int p = 0; p < P; ++p)
        {
            float4 plane = plane_d[cc + (p + s_c) * num_clus];

            float D_f = plane.x * x + plane.y * y + plane.z;
            float cost = abs(D - D_f) <= inlier_thrsh ? 0 : 1;
            im_plane_cost_d[id + p * im_size] = lambda * cost;
        }
    }
    else
    {
        int cc = im_D4_d[id].x;
        int im_id = im_D4_d[id].z;
        int x = im_id % width;
        int y = im_id / width;
        for (int p = 0; p < P; ++p)
        {
            float cost = 0;
            float4 plane = plane_d[cc + (p + s_c) * num_clus];

            float d_f = plane.x * x + plane.y * y + plane.z;
            int d = __float2int_rd(d_f);
            int d_ = d + 1;
            if (d < 0 || d_ >= m_labels)
                cost = tau;
            else
            {
                cost = (d_ - d_f) * im_cost_d[im_id + d * im_size] + (d_f - d) * im_cost_d[im_id + d_ * im_size];
            }
            im_plane_cost_d[id + p * im_size] = lambda * cost;
        }
    }
}

__global__ void
plane_inlier_ratio(const float4 *const plane_d, const float4 *const im_D4_d,
                   float *const im_plane_cost_d, const int im_size,
                   const int width, const float inlier_thrsh,
                   const int num_clus, const int K)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= im_size)
        return;

    int cc = im_D4_d[id].x;
    float D = im_D4_d[id].y;
    int im_id = im_D4_d[id].z;
    int x = im_id % width;
    int y = im_id / width;
    for (int k = 0; k < K; ++k)
    {
        float4 plane = plane_d[cc + k * num_clus];

        float D_f = plane.x * x + plane.y * y + plane.z;
        float is_inlier = abs(D - D_f) <= inlier_thrsh ? 1 : 0;
        im_plane_cost_d[id + k * im_size] = is_inlier;
    }
}

__global__ void
set_occlusion_cost_k(const float *const inlier_ratio_d,
                     const float *const mean_d, float *const cost_d,
                     const float inlier_ratio, const float tau,
                     const int num_clus, const int mean_channels, const int K)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    float w = mean_d[id + (mean_channels - 1) * num_clus];
    for (int k = 8; k < K; ++k)
    {
        if (inlier_ratio_d[id + k * num_clus] / w < inlier_ratio)
            cost_d[id + k * num_clus] = tau * w;
    }
}

__global__ void
normalize_msg_k(float *const msg_d, const int size, const int K)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size)
        return;

    float mean = 0;
    for (int k = 0; k < K; ++k)
    {
        mean += msg_d[id + k * size];
    }
    mean /= K;
    for (int k = 0; k < K; ++k)
    {
        msg_d[id + k * size] -= mean;
    }
}

__global__ void
passing_msg_k(const int2 *const bd_d, const float *const belief_d,
              float *const msg_d, const int num_bd, const int num_clus,
              const int P)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= num_bd || y >= P * 2)
        return;

    int t = bd_d[x].y;
    msg_d[x + y * num_bd] = belief_d[t + y * num_clus] - msg_d[x + y * num_bd];
}

__global__ void
smooth_msg_k(const int2 *const bd_d, const float *const dc_d,
             const int *const bdl_d, const int4 *const point_id_d,
             const float4 *const plane_d, const float *const msg_d,
             float *const msg_t_d, const float m_tau_smooth, const int num_bd, const int num_clus,
             const int width, const int P)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= num_bd || y >= P * 3)
        return;

    int2 bd = bd_d[x];
    int s = bd.x, t = bd.y;
    float4 plane_s = plane_d[s + y * num_clus];
    int4 p_s = point_id_d[s];
    int4 p_t = point_id_d[t];
    float dist_color = dc_d[x];
    float dist_pos = dc_d[x + num_bd];
    int bdl = bdl_d[x];
    dist_pos = max(dist_pos, 1.0f);
    dist_color = max(dist_color, 0.01f);
    float c0 = dist_color * bdl;
    float minimum = FLT_MAX;
    msg_t_d[x + y * num_bd] = 0;
    for (int p = 0; p < P * 2; ++p)
    {
        if (p == P)
        {
            msg_t_d[x + y * num_bd] += minimum;
            minimum = FLT_MAX;
        }
        float4 plane_t = plane_d[t + p * num_clus];
        float delta_px = plane_s.x - plane_t.x;
        float delta_py = plane_s.y - plane_t.y;
        float delta_pz = plane_s.z - plane_t.z;
        float phi = abs(delta_px * (p_s.x % width) + delta_py * (p_s.x / width) + delta_pz) +
                    abs(delta_px * (p_s.y % width) + delta_py * (p_s.y / width) + delta_pz) +
                    abs(delta_px * (p_s.z % width) + delta_py * (p_s.z / width) + delta_pz) +
                    abs(delta_px * (p_t.x % width) + delta_py * (p_t.x / width) + delta_pz) +
                    abs(delta_px * (p_t.y % width) + delta_py * (p_t.y / width) + delta_pz) +
                    abs(delta_px * (p_t.z % width) + delta_py * (p_t.z / width) + delta_pz);

        phi = min(phi / 3 / dist_pos, m_tau_smooth);
        float msg = msg_d[x + p * num_bd] + c0 * phi;
        if (msg < minimum)
            minimum = msg;
    }
    msg_t_d[x + y * num_bd] += minimum;
}

__global__ void
smooth_msg(const int2 *const bd_d, const float *const dc_d,
           const int *const bdl_d, const float *const mean_d,
           const float4 *const plane_d, const float *const msg_d,
           float *const msg_t_d, const float m_tau_smooth, const int num_bd, const int num_clus,
           const int P)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= num_bd || y >= P * 3)
        return;

    int2 bd = bd_d[x];
    int s = bd.x, t = bd.y;
    float4 plane_s = plane_d[s + y * num_clus];
    float x0 = mean_d[s + 3 * num_clus];
    float y0 = mean_d[s + 4 * num_clus];
    float x1 = mean_d[t + 3 * num_clus];
    float y1 = mean_d[t + 4 * num_clus];
    float dist_color = dc_d[x];
    float dist_pos = dc_d[x + num_bd];
    int bdl = bdl_d[x];
    dist_pos = max(dist_pos, 1.0f);
    dist_color = dist_color > 0.01f ? dist_color : 0.01f;
    float c0 = dist_color * bdl;
    float minimum = FLT_MAX;
    msg_t_d[x + y * num_bd] = 0;
    for (int p = 0; p < P * 2; ++p)
    {
        if (p == P)
        {
            msg_t_d[x + y * num_bd] += minimum;
            minimum = FLT_MAX;
        }
        float4 plane_t = plane_d[t + p * num_clus];
        float delta_px = plane_s.x - plane_t.x;
        float delta_py = plane_s.y - plane_t.y;
        float delta_pz = plane_s.z - plane_t.z;
        float phi = abs(delta_px * x0 + delta_py * y0 + delta_pz) +
                    abs(delta_px * x1 + delta_py * y1 + delta_pz);
        phi = min(phi / dist_pos, m_tau_smooth);
        float msg = msg_d[x + p * num_bd] + c0 * phi;
        if (msg < minimum)
            minimum = msg;
    }
    msg_t_d[x + y * num_bd] += minimum;
}

__global__ void
aggregate_belief_k(const int2 *const bd_d, const float *const msg_t_d,
                   float *const belief_d, const int num_bd, const int num_clus,
                   const int K)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= num_bd || y >= K)
        return;

    int s = bd_d[x].x;
    atomicAdd(&belief_d[s + y * num_clus], msg_t_d[x + y * num_bd]);
}

__global__ void
ping_pang_msg_k(float *const msg_d, const float *const msg_t_d,
                const int *const reverse_id_d, const int num_bd, const int K)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= num_bd || y >= K)
        return;

    msg_d[reverse_id_d[x] + y * num_bd] = msg_t_d[x + y * num_bd];
}

// TODO: this is time consuming
__global__ void
get_belief_order(const float *const belief_d, int *const order_d,
                 const int num_clus, const int K)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    for (int i = 0; i < K - 1; ++i)
    {
        for (int j = i + 1; j < K; ++j)
        {
            if (belief_d[id + j * num_clus] < belief_d[id + i * num_clus])
                order_d[id + i * num_clus]++;
            else
                order_d[id + j * num_clus]++;
        }
    }
}

__global__ void
select_belief(const int *const order_d, float4 *const plane_d,
              float *const cost_d, float *const belief_d, const int num_clus,
              const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    int num_best = 0;
    for (int k = 0; k < P * 3; ++k)
    {
        if (order_d[id + k * num_clus] < P)
        {
            plane_d[id + num_best * num_clus] = plane_d[id + k * num_clus];
            cost_d[id + num_best * num_clus] = cost_d[id + k * num_clus];
            belief_d[id + num_best * num_clus] = belief_d[id + k * num_clus];
            num_best++;
        }
    }
}

__global__ void
select_msg(const int2 *const bd_d, const int *const order_d,
           float *const msg_d, const int num_bd, const int num_clus,
           const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_bd)
        return;

    int t = bd_d[id].y;
    int num_best = 0;
    for (int k = 0; k < P * 3; ++k)
    {
        if (order_d[t + k * num_clus] < P)
        {
            msg_d[id + num_best * num_bd] = msg_d[id + k * num_bd];
            num_best++;
        }
    }
}

__forceinline__ __device__ void
normalize_plane(float4 &plane)
{
    float norm = sqrtf(
        plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
    plane.x /= norm;
    plane.y /= norm;
    plane.z /= norm;
}
__forceinline__ __device__ void
make_three_diff_id(int &p0, int &p1, int &p2, const int &tn)
{
    if (tn <= 3)
    {
        p1 = (p0 + 1) % tn;
        p2 = (p1 + 1) % tn;
        return;
    }
    if (p1 == p0)
        p1 = tn - 1;
    if (p2 == p0)
    {
        if (p1 == tn - 2)
            p2 = tn - 1;
        else
            p2 = tn - 2;
    }
    else if (p2 == p1)
    {
        if (p0 == tn - 2)
            p2 = tn - 1;
        else
            p2 = tn - 2;
    }
}
// A, B, C are in x, y ,z, w format
__forceinline__ __device__ float4
fit_plane_from_points(const float4 &A, const float4 &B, const float4 &C)
{
    float4 a = B - A;
    float4 b = C - A;
    float nx = a.y * b.z - a.z * b.y;
    float ny = a.z * b.x - a.x * b.z;
    float nz = a.x * b.y - a.y * b.x;
    nx /= nz;
    ny /= nz;
    nz /= nz;
    if (abs(nz - 1.f) > FLT_EPSILON)
    {
        return make_float4(0.f, 0.f, (A.z + B.z + C.z) / 3.f, 0.f);
    }
    else
        return make_float4(-nx, -ny, nx * A.x + ny * A.y + A.z, 0.f);
}

// linear congruential generator
__device__ int
rand_lcg(unsigned int s)
{
    // BSD_RND
    int a = 1103515245;
    int c = 12345;
    unsigned int m = 2147483648;
    return (a * s + c) % m;
}
__device__ float
rand_lcg_uniform(unsigned int &s, const float &f)
{
    // BSD_RND
    int a = 1103515245;
    int c = 12345;
    unsigned int m = 2147483648;
    s = (a * s + c) % m;
    return s / (float)m * f;
}
__global__ void
init_rand_seed_k(int *const s_child_d, int *const s_neighbor_d,
                 int *const s_ransac_d, const int *const clus_num_pos_d,
                 const int *const nb_num_pos_d,
                 const int *const im_clus_num_pos_d, const int num_clus,
                 const int num_clus_h)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    // s_neighbor = nb_pos
    s_neighbor_d[id] = nb_num_pos_d[id + num_clus + 1];
    // s_ransac = clus_pos
    s_ransac_d[id] = im_clus_num_pos_d[id + num_clus];

    if (id >= num_clus_h)
        return;

    // s_child = clus_pos
    s_child_d[id] = clus_num_pos_d[id + num_clus_h];
}
__global__ void
ransac_candidate_plane(int *const s_d, const float4 *const im_D4_d,
                       const int *const im_clus_num_pos_d,
                       float4 *const plane_d, int4 *const point_id_d,
                       const int num_clus, const int width, const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    int clus_num = im_clus_num_pos_d[id];
    int clus_pos = im_clus_num_pos_d[id + num_clus];
    unsigned int s = s_d[id];

    for (int p = P * 2; p < P * 3; ++p)
    {
        s = rand_lcg(s);
        int p0 = s % clus_num;
        s = rand_lcg(s);
        int p1 = s % (clus_num - 1);
        s = rand_lcg(s);
        int p2 = s % (clus_num - 2);
        make_three_diff_id(p0, p1, p2, clus_num);
        p0 += clus_pos;
        p1 += clus_pos;
        p2 += clus_pos;

        float4 P0 = im_D4_d[p0];
        float4 P1 = im_D4_d[p1];
        float4 P2 = im_D4_d[p2];
        int p0_id = P0.z;
        int p1_id = P1.z;
        int p2_id = P2.z;

        float4 A = make_float4(p0_id % width, p0_id / width, P0.y, 0.f);
        float4 B = make_float4(p1_id % width, p1_id / width, P1.y, 0.f);
        float4 C = make_float4(p2_id % width, p2_id / width, P2.y, 0.f);

        plane_d[id + p * num_clus] = fit_plane_from_points(A, B, C);

        if (p == P * 3 - 1)
        {
            point_id_d[id].x = p0_id;
            point_id_d[id].y = p1_id;
            point_id_d[id].z = p2_id;
        }
    }
    s_d[id] = s;
}
__global__ void
random_candidate_plane(int *const s_d, const float *const mean_d,
                       float4 *const plane_d, const float max_dz,
                       const float max_dn, const int num_clus, const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    unsigned int s = s_d[id];
    float x = mean_d[id + 3 * num_clus];
    float y = mean_d[id + 4 * num_clus];
    for (int p = P * 2; p < P * 3; ++p)
    {
        float4 plane = plane_d[id + p * num_clus];
        float z = plane.x * x + plane.y * y + plane.z;
        // TODO: try normal distribution
        float delta_z = rand_lcg_uniform(s, max_dz);
        z += delta_z;
        float delta_nx = rand_lcg_uniform(s, max_dn);
        float delta_ny = rand_lcg_uniform(s, max_dn);
        float delta_nz = rand_lcg_uniform(s, max_dn);
        normalize_plane(plane);
        plane.x += delta_nx;
        plane.y += delta_ny;
        plane.z += delta_nz;
        // normalize_plane(plane);
        float a = -plane.x / plane.z;
        float b = -plane.y / plane.z;
        float c = z - a * x - b * y;
        plane_d[id + p * num_clus] = make_float4(a, b, c, 0.f);
    }
    s_d[id] = s;
}

__global__ void
neighbor_candidate_plane(int *const s_d, const int2 *const bd_d,
                         const int *const nb_num_pos_d,
                         const float4 *const plane_d,
                         float4 *const plane_buf_d, const int num_clus,
                         const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    int nb_num = nb_num_pos_d[id];
    int nb_pos = nb_num_pos_d[id + num_clus + 1];
    unsigned int s = s_d[id];
    for (int p = 0; p < P; ++p)
    {
        /*
        // This is more time-consuming
         float r_pos = (1 - curand_uniform (&state)) * nb_num;
         int t_id = bd_d[__float2int_rd (r_pos) + nb_pos].y;
         */
        s = rand_lcg(s);
        int r_pos = s % nb_num;
        int t_id = bd_d[r_pos + nb_pos].y;
        plane_buf_d[id + p * num_clus] = plane_d[t_id + (p + P * 2) * num_clus];
    }
    s_d[id] = s;
}

__global__ void
select_child_plane(int *const s_d, const int *const clus_sort_d,
                   const int *const clus_num_pos_d,
                   const float4 *const plane_d, float4 *const plane_h_d,
                   const int num_clus, const int num_clus_h, const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus_h)
        return;

    int clus_num = clus_num_pos_d[id];
    int clus_pos = clus_num_pos_d[id + num_clus_h];
    unsigned int s = s_d[id];
    for (int p = 0; p < P; ++p)
    {
        s = rand_lcg(s);
        int r_pos = s % clus_num;
        int c = clus_sort_d[r_pos + clus_pos];
        plane_h_d[id + (p + P * 2) * num_clus_h] = plane_d[c];
    }
    s_d[id] = s;
}
__global__ void
set_candidate_plane_k(const int *const clus_d, const float4 *const plane_h_d,
                      float4 *const plane_d, const int num_clus,
                      const int num_clus_h, const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    int c = clus_d[id];
    for (int p = P * 2; p < P * 3; ++p)
    {
        plane_d[id + p * num_clus] = plane_h_d[c + p * num_clus_h];
    }
}

__global__ void
compute_plane_confidence_k(float *const plane_cfd_d, const float *const mean_d, const float *const cost_d, const int num_clus, const int mean_channels, const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    float w = mean_d[id + (mean_channels - 1) * num_clus];
    for (int p = 0; p < P * 3; ++p)
    {
        float avg_cost = cost_d[id + p * num_clus] / w;
        plane_cfd_d[id + p * num_clus] = 1 - avg_cost;
    }
}

__global__ void
reset_candidate_plane_k(float4 *const plane_d, const float *const plane_cfd_d, const int num_clus, const int P)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_clus)
        return;

    float4 sum_plane = make_float4(0, 0, 0, 0);
    float sum_plane_cfd = 0;
    for (int p = 0; p < P * 3; ++p)
    {
        float cfd = plane_cfd_d[id + p * num_clus];
        if (cfd > 0.6)
        {
            sum_plane += plane_d[id + p * num_clus] * cfd;
            sum_plane_cfd += cfd;
        }
    }
    for (int p = P * 2; p < P * 3; ++p)
    {
        if (plane_cfd_d[id + p * num_clus] < 0.4 && sum_plane_cfd > 0.6)
            plane_d[id + p * num_clus] = sum_plane / sum_plane_cfd;
    }
}

__global__ void
map_msg_k(const float *const belief_d, float *const min_belief_d, const int N,
          const int C)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= N || y >= C)
        return;

    int id = y * N + x;
    atomicMinFloat(&min_belief_d[x], belief_d[id]);
}

__global__ void
min_belief_to_plane(const float *const belief_d,
                    const float *const min_belief_d,
                    const float4 *const plane_d, float4 *const min_plane_d,
                    const int num_clus, const int P)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_clus)
        return;

    int min_p = 0;
    for (int p = 0; p < P; ++p)
    {
        if (belief_d[id + p * num_clus] == min_belief_d[id])
        {
            min_p = p;
            break;
        }
    }

    min_plane_d[id] = plane_d[id + min_p * num_clus];
}

__global__ void
plane_to_label(const float4 *const im_D4_d,
               const float4 *const min_plane_d, float *const disp_img_d,
               float *const label_img_d, const int im_size, const int width,
               const int m_labels)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= im_size)
        return;

    int c = im_D4_d[id].x;
    int im_id = im_D4_d[id].z;
    int x = im_id % width;
    int y = im_id / width;
    float4 plane = min_plane_d[c];
    float d = plane.x * x + plane.y * y + plane.z;
    d = fminf(fmaxf(d, 0), m_labels - 1);
    disp_img_d[im_id] = d;
    // label
    label_img_d[im_id * 3] = plane.x;
    label_img_d[im_id * 3 + 1] = plane.y;
    label_img_d[im_id * 3 + 2] = plane.z;
}

__global__ void
set_D4_k(const int *const im_clus_d, const float *const im_D_d,
         float4 *const im_D4_d, const int im_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= im_size)
        return;

    im_D4_d[id].x = im_clus_d[id];
    im_D4_d[id].y = im_D_d[id];
    im_D4_d[id].z = id;
    // im_D4_d[id].w is the depth cluster id.
}

//-------------------------------------------------------
float HBP_ISP::set_lambda_by_KL_divergence(const float *const im_cost_d, float *const KL_d)
{
    switch_level(0); // pixel level
    assert(num_clus = im_size);
    cudaMemset(KL_d, 0, sizeof(float) * num_bd);
    int grid = (num_bd + m_block - 1) / m_block;
    compute_KL_k<<<grid, m_block>>>(im_cost_d, bd_d, KL_d, num_clus, m_labels, num_bd);

    float lambda = m_alpha / (thrust::reduce(thrust::device, KL_d, KL_d + num_bd) / num_bd);
    std::cout << "lambda: " << lambda << std::endl;
    return lambda;
}

void HBP_ISP::setup_D4(int **isp_im_clus_d, float4 **isp_im_D4_d)
{
    int grid = (im_size + m_block - 1) / m_block;
    // init im_clus
    set_to_identity<<<grid, m_block>>>(isp_im_clus_d[0], im_size);
    for (int i = 0; i < isp_levels - 1; ++i)
    {
        update_image_label<<<grid, m_block>>>(
            isp_clus_d[i],
            isp_im_clus_d[i], isp_im_clus_d[i + 1], im_size);
    }

    for (int i = 0; i < isp_levels; ++i)
    {
        set_D4_k<<<grid, m_block>>>(isp_im_clus_d[i], im_D_d, isp_im_D4_d[i],
                                    im_size);
        // sort_by_key
        thrust::sort_by_key(thrust::device, isp_im_clus_d[i],
                            isp_im_clus_d[i] + im_size, isp_im_D4_d[i]);
    }
}

void HBP_ISP::switch_level(const int level)
{
    // set all pointers to current level, so that we can reference them easily
    clus_d = isp_clus_d[level];
    im_clus_d = isp_im_clus_d[level];
    mean_d = isp_mean_d[level];
    cost_d = isp_cost_d[level];
    belief_d = isp_belief_d[level];
    plane_d = isp_plane_d[level];
    bd_d = isp_bd_d[level];
    dc_d = isp_dc_d[level];
    bdl_d = isp_bdl_d[level];
    clus_sort_d = isp_clus_sort_d[level];
    clus_num_pos_d = isp_clus_num_pos_d[level];
    im_clus_num_pos_d = isp_im_clus_num_pos_d[level];
    nb_num_pos_d = isp_nb_num_pos_d[level];
    reverse_id_d = isp_reverse_id_d[level];
    im_D4_d = isp_im_D4_d[level];
    point_id_d = isp_point_id_d[level];
    num_clus = num_clus_isp[level];
    num_bd = num_bd_isp[level];
    if (level < isp_levels - 1)
    {
        im_clus_h_d = isp_im_clus_d[level + 1];
        mean_h_d = isp_mean_d[level + 1];
        cost_h_d = isp_cost_d[level + 1];
        belief_h_d = isp_belief_d[level + 1];
        plane_h_d = isp_plane_d[level + 1];
        num_clus_h = num_clus_isp[level + 1];
    }
    else
    {
        num_clus_h = 1;
    }
    m_level = level;
}

void HBP_ISP::init_bp(float4 *const plane_d, float *const cost_d,
                      float *const belief_d)
{
    cudaMemset(msg_d, 0, sizeof(float) * num_bd * K);

    int grid = (num_clus + m_block - 1) / m_block;

    copy_prior_plane<<<grid, m_block>>>(clus_d, plane_h_d, plane_d, num_clus,
                                        num_clus_h, P);

    compute_plane_cost(0);
    compute_plane_cost(P);

    // belief_d = cost_d, since msg_d = zero
    cudaMemcpy(belief_d, cost_d, sizeof(float) * num_clus * K,
               cudaMemcpyDeviceToDevice);

    init_rand_seed_k<<<grid, m_block>>>(s_child_d, s_neighbor_d, s_ransac_d,
                                        clus_num_pos_d, nb_num_pos_d, im_clus_num_pos_d, num_clus,
                                        num_clus_h);
}

void HBP_ISP::compute_dist_color_pos(float *const mean_d, float *const dc_d)
{
    // set to zero
    cudaMemset(dc_d, 0, sizeof(float) * num_bd * dc_channels);

    // normalize
    dim3 grids((num_clus + m_blocks.x - 1) / m_blocks.x,
               (mean_channels + m_blocks.y - 1) / m_blocks.y);
    sum_to_mean<<<grids, m_blocks>>>(mean_d, num_clus, mean_channels);

    grids = dim3((num_bd + m_blocks.x - 1) / m_blocks.x,
                 (mean_channels - 1 + m_blocks.y - 1) / m_blocks.y);
    compute_dist_color_pos_k<<<grids, m_blocks>>>(
        mean_d,
        bd_d, dc_d, num_bd, num_clus, mean_channels - 1);

    grids.y = (dc_channels + m_blocks.y - 1) / m_blocks.y;
    dist_trans<<<grids, m_blocks>>>(dc_d, m_gamma, num_bd, dc_channels);
}

void HBP_ISP::reduce_num_pos(const int *const key_d, int *num_pos_d,
                             const int src_size, const int dst_size)
{
    int grid = (src_size + m_block - 1) / m_block;
    cudaMemset(num_pos_d, 0, sizeof(int) * dst_size);
    // since we use atomicAdd, clus_sort_d and clus_d both works
    sum_reduce_num_clus<<<grid, m_block>>>(
        key_d,
        num_pos_d, src_size);
    thrust::exclusive_scan(thrust::device, num_pos_d, num_pos_d + dst_size,
                           num_pos_d + dst_size);
}

void HBP_ISP::sort_clus(int *const clus_sort_d, int *const clus_num_pos_d,
                        int *const im_clus_num_pos_d)
{
    // clus_sort_d shall be key + clus form
    int grid = (num_clus + m_block - 1) / m_block;
    set_to_identity<<<grid, m_block>>>(clus_sort_d, num_clus);
    cudaMemcpy(clus_sort_d + num_clus, clus_d, sizeof(int) * num_clus,
               cudaMemcpyDeviceToDevice);
    thrust::sort_by_key(thrust::device, clus_sort_d + num_clus,
                        clus_sort_d + num_clus * 2, clus_sort_d);

    // since we use atomicAdd, clus_sort_d and clus_d both works
    reduce_num_pos(clus_sort_d + num_clus, clus_num_pos_d, num_clus,
                   num_clus_h);

    reduce_num_pos(im_clus_d, im_clus_num_pos_d, im_size, num_clus);
}

void HBP_ISP::reverse_bd_id(const int2 *const bd_d, int *const reverse_id_d)
{
    int grid = (num_bd + m_block - 1) / m_block;

    // num of neighbors for each cluster
    cudaMemset(nb_num_pos_d, 0, sizeof(int) * (num_clus + 1));
    sum_reduce_num_nbs<<<grid, m_block>>>(bd_d, nb_num_pos_d, num_bd);

    // exclusive scan
    thrust::exclusive_scan(thrust::device, nb_num_pos_d,
                           nb_num_pos_d + num_clus + 1,
                           nb_num_pos_d + num_clus + 1);

    reverse_bd_id_k<<<grid, m_block>>>(
        bd_d,
        nb_num_pos_d + num_clus + 1, reverse_id_d, num_bd);
}

void HBP_ISP::passing_msg(const float4 *const plane_d, const float *const belief_d,
                          float *const msg_d, float *const msg_t_d)
{
    // compute 3P candidate labels simultaneously and select P minimum
    // - such that their beliefs are computed using the same messages
    // msg_t_d = belief_d - msg_d
    // smooth msg_t_d
    int grid = (num_bd + m_block - 1) / m_block;
    dim3 grids((num_bd + m_blocks.x - 1) / m_blocks.x,
               (K + m_blocks.y - 1) / m_blocks.y);
    passing_msg_k<<<grids, m_blocks>>>(
        bd_d,
        belief_d, msg_d, num_bd, num_clus, P);

    // TODO: perform the smoothness test on the three chosen points
    //               of each superpixel.
    smooth_msg_k<<<grids, m_blocks>>>(bd_d, dc_d, bdl_d, point_id_d, plane_d,
                                      msg_d, msg_t_d, m_tau_smooth, num_bd, num_clus, width, P);
    /*
    // This is the normal smoothness regularization
     smooth_msg<<<grids, m_blocks>>>(
     bd_d,
     dc_d, bdl_d, mean_d, plane_d, msg_d, msg_t_d, m_tau_smooth, num_bd, num_clus,
     P);
     */
    normalize_msg_k<<<grid, m_block>>>(msg_t_d, num_bd, K);
}
void HBP_ISP::aggregate_belief(const float *const msg_t_d,
                               const float *const cost_d, float *const belief_d)
{
    cudaMemcpy(belief_d, cost_d, sizeof(float) * num_clus * K,
               cudaMemcpyDeviceToDevice);
    // belief_d += all msg_t_d
    dim3 grids((num_bd + m_blocks.x - 1) / m_blocks.x,
               (K + m_blocks.y - 1) / m_blocks.y);
    aggregate_belief_k<<<grids, m_blocks>>>(bd_d, msg_t_d, belief_d, num_bd,
                                            num_clus, K);
}
void HBP_ISP::ping_pang_msg(float *const msg_d, const float *const msg_t_d)
{
    dim3 grids((num_bd + m_blocks.x - 1) / m_blocks.x,
               (K + m_blocks.y - 1) / m_blocks.y);
    ping_pang_msg_k<<<grids, m_blocks>>>(
        msg_d,
        msg_t_d, reverse_id_d, num_bd, K);
}
void HBP_ISP::select_candidate_plane(float4 *const plane_d, float *const cost_d,
                                     float *const belief_d, float *const msg_d)
{
    int grid = (num_clus + m_block - 1) / m_block;
    cudaMemset(order_d, 0, sizeof(int) * num_clus * K);
    get_belief_order<<<grid, m_block>>>(belief_d, order_d, num_clus, K);
    select_belief<<<grid, m_block>>>(
        order_d,
        plane_d, cost_d, belief_d, num_clus, P);

    grid = (num_bd + m_block - 1) / m_block;
    select_msg<<<grid, m_block>>>(bd_d, order_d, msg_d, num_bd, num_clus, P);
}

std::pair<cv::Mat, cv::Mat>
HBP_ISP::map_msg(const float4 *const plane_d, const float *const belief_d)
{
    int grid = (num_clus + m_block - 1) / m_block;
    dim3 grids((num_clus + m_blocks.x - 1) / m_blocks.x,
               (P + m_blocks.y - 1) / m_blocks.y);

    set_to_zero<<<grid, m_block>>>(min_belief_d, num_clus, FLT_MAX);
    map_msg_k<<<grids, m_blocks>>>(belief_d, min_belief_d, num_clus, P);

    min_belief_to_plane<<<grid, m_block>>>(belief_d, min_belief_d, plane_d,
                                           min_plane_d, num_clus, P);

    grid = (im_size + m_block - 1) / m_block;
    plane_to_label<<<grid, m_block>>>(im_D4_d, min_plane_d, disp_img_d,
                                      label_img_d, im_size, width, m_labels);

    cv::Mat disp_img_h(height, width, CV_32FC1);
    cudaMemcpy(disp_img_h.data, disp_img_d, sizeof(float) * im_size,
               cudaMemcpyDeviceToHost);
    cv::Mat label_img_h(height, width, CV_32FC3);
    cudaMemcpy(label_img_h.data, label_img_d, sizeof(float) * im_size * 3,
               cudaMemcpyDeviceToHost);
    return {disp_img_h, label_img_h};
}

cv::Mat HBP_ISP::draw_spx_mean(const float *const mean_d)
{
    cv::Mat spx_mean(height, width, CV_32FC3);
    float4 *im_D4_h = new float4[im_size];
    float *mean_h = new float[num_clus * mean_channels];
    int2 *bd_h = new int2[num_bd];
    cudaMemcpy(im_D4_h, im_D4_d, sizeof(float4) * im_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mean_h, mean_d, sizeof(float) * num_clus * mean_channels, cudaMemcpyDeviceToHost);
    cudaMemcpy(bd_h, bd_d, sizeof(int2) * num_bd, cudaMemcpyDeviceToHost);
    // first draw spx mean
    for (int i = 0; i < im_size; ++i)
    {
        float4 im_D4 = im_D4_h[i];
        int im_clus = im_D4.x;
        int im_id = im_D4.z;
        int x = im_id % width;
        int y = im_id / width;
        spx_mean.at<cv::Vec3f>(y, x) = cv::Vec3f(mean_h[im_clus], mean_h[im_clus + num_clus], mean_h[im_clus + num_clus * 2]);
    }
    cv::cvtColor(spx_mean, spx_mean, cv::COLOR_Lab2BGR);
    spx_mean.convertTo(spx_mean, CV_8UC3, 255);
    if (m_level >= 2)
    {
        // then draw MRF network
        for (int i = 0; i < num_clus; ++i)
        { // draw nodes
            float x = mean_h[i + num_clus * 3];
            float y = mean_h[i + num_clus * 4];
            cv::circle(spx_mean, cv::Point2f(x, y), 5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
        for (int i = 0; i < num_bd; ++i)
        {
            int s = bd_h[i].x;
            int t = bd_h[i].y;
            float s_x = mean_h[s + num_clus * 3];
            float s_y = mean_h[s + num_clus * 4];
            float t_x = mean_h[t + num_clus * 3];
            float t_y = mean_h[t + num_clus * 4];
            cv::line(spx_mean, cv::Point2f(s_x, s_y), cv::Point2f(t_x, t_y), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }

    delete[] bd_h;
    delete[] mean_h;
    delete[] im_D4_h;

    return spx_mean;
}

/********************************************************************************/

// TODO: use grids (im_size, P) may have speed up
void HBP_ISP::compute_plane_cost(const int s_c)
{
    // compute cost of candidate labels
    int grid = (im_size + m_block - 1) / m_block;
    read_plane_cost<<<grid, m_block>>>(plane_d, im_D4_d, im_cost_d,
                                       im_plane_cost_d, use_cost_vol, m_tau, im_size, width,
                                       m_labels, inlier_thrsh, m_lambda, num_clus, s_c, P);
    // reduce plane cost
    for (int p = 0; p < P; ++p)
    {
        thrust::reduce_by_key(thrust::device, im_clus_d, im_clus_d + im_size,
                              im_plane_cost_d + p * im_size,
                              thrust::make_discard_iterator(),
                              cost_d + (p + s_c) * num_clus);
    }
}

void HBP_ISP::compute_inlier_ratio(float *const inlier_ratio_d)
{
    int grid = (im_size + m_block - 1) / m_block;
    plane_inlier_ratio<<<grid, m_block>>>(plane_d, im_D4_d, im_plane_cost_d,
                                          im_size, width, inlier_thrsh, num_clus, K);
    // reduce plane cost
    for (int k = 0; k < K; ++k)
    {
        thrust::reduce_by_key(thrust::device, im_clus_d, im_clus_d + im_size,
                              im_plane_cost_d + k * im_size,
                              thrust::make_discard_iterator(),
                              inlier_ratio_d + k * num_clus);
    }
}

void HBP_ISP::set_occlusion_cost(const float *const inlier_ratio_d,
                                 float *const cost_d)
{
    int grid = (num_clus + m_block - 1) / m_block;
    set_occlusion_cost_k<<<grid, m_block>>>(inlier_ratio_d, mean_d, cost_d,
                                            inlier_ratio, m_tau, num_clus, mean_channels, K);
}

void HBP_ISP::occlusion_test(const int iter_bp)
{
    compute_inlier_ratio(inlier_ratio_d);

    set_occlusion_cost(inlier_ratio_d, cost_d);
}

void HBP_ISP::belief_propagation(float4 *const plane_d, float *const cost_d,
                                 float *const belief_d, float *const msg_d)
{
    // perform normal BP in lower level
    passing_msg(plane_d, belief_d, msg_d, msg_t_d);
    aggregate_belief(msg_t_d, cost_d, belief_d);
    ping_pang_msg(msg_d, msg_t_d);
    select_candidate_plane(plane_d, cost_d, belief_d, msg_d);
}

bool HBP_ISP::generate_plane(int &iter)
{
    if (iter <= iter_ransac_search)
    { // random search
        int grid = (num_clus + m_block - 1) / m_block;
        ransac_candidate_plane<<<grid, m_block>>>(s_ransac_d, im_D4_d,
                                                  im_clus_num_pos_d, plane_d, point_id_d, num_clus,
                                                  width, P);
    }
    else if (iter <= iter_spatial_prop)
    {
        // randomly select a child plane as parent's candidate
        int grid = (num_clus_h + m_block - 1) / m_block;
        select_child_plane<<<grid, m_block>>>(s_child_d, clus_sort_d,
                                              clus_num_pos_d, plane_d, plane_h_d, num_clus,
                                              num_clus_h, P);
        // perform spatial prop and random search in parent level
        switch_level(m_level + 1);
        neighbor_candidate_plane<<<grid, m_block>>>(s_neighbor_d, bd_d,
                                                    nb_num_pos_d, plane_d, plane_buf_d, num_clus,
                                                    P);
        cudaMemcpy(plane_d + P * 2 * num_clus, plane_buf_d,
                   sizeof(float4) * P * num_clus, cudaMemcpyDeviceToDevice);
        switch_level(m_level - 1);
        grid = (num_clus + m_block - 1) / m_block;
        set_candidate_plane_k<<<grid, m_block>>>(clus_d, plane_h_d, plane_d,
                                                 num_clus, num_clus_h, P);
    }
    else if (iter <= iter_random_adjust && max_dz >= end_dz)
    {
        int grid = (num_clus + m_block - 1) / m_block;
        random_candidate_plane<<<grid, m_block>>>(
            s_ransac_d,
            mean_d, plane_d, max_dz, max_dn, num_clus, P);
        max_dz /= 2.0f;
        max_dn /= 2.0f;
    }
    else
    {
        return false;
    }

    // child and parent have the same candidate labels
    iter++;
    return true;
}

void HBP_ISP::compute_plane_confidence(float *const plane_cfd_d)
{
    int grid = (num_clus + m_block - 1) / m_block;
    compute_plane_confidence_k<<<grid, m_block>>>(plane_cfd_d, mean_d, cost_d, num_clus, mean_channels, P);
}

void HBP_ISP::reset_candidate_plane(float4 *const plane_d, const float *const plane_cfd_d)
{
    int grid = (num_clus + m_block - 1) / m_block;
    reset_candidate_plane_k<<<grid, m_block>>>(plane_d, plane_cfd_d, num_clus, P);
}

void HBP_ISP::candidate_test(float4 *const plane_d)
{
    compute_plane_cost(P * 2);
    compute_plane_confidence(plane_cfd_d);
    reset_candidate_plane(plane_d, plane_cfd_d);
}


void HBP_ISP::hbp_g(float4 *const plane_d, float *const belief_d)
{
    // Set best = prior = higher level best
    // init all msg to zero
    init_bp(plane_d, cost_d, belief_d);

    const int ITER = 10;
    int iter_bp = ITER + m_level * ITER;
    max_dz = m_labels / 2.0f * m_level / isp_levels;
    max_dn = 1.0f * m_level / isp_levels;
    for (int i = 0; i < iter_bp; ++i)
    {
        iter_ransac_search = 3;
        iter_spatial_prop = 1 + iter_ransac_search;
        iter_random_adjust = 1 + iter_spatial_prop;
        if (i <= iter_bp / 4 * 3)
            iter_random_adjust = iter_spatial_prop;
        m_plane_iter = 0;

        while (generate_plane(m_plane_iter))
        {
            // candidate_test(plane_d);
            compute_plane_cost(P * 2);

            belief_propagation(plane_d, cost_d, belief_d, msg_d);
        }

    }
}

std::pair<cv::Mat, cv::Mat>
HBP_ISP::run_ms(const int min_level)
{
    std::cout << "start HBP-ISP" << std::endl;
    // TODO: class depth to similar groups. write to D4

    for (int i = 0; i < isp_levels; ++i)
    {
        switch_level(i);
        reverse_bd_id(bd_d, reverse_id_d);
        compute_dist_color_pos(mean_d, dc_d);
        sort_clus(clus_sort_d, clus_num_pos_d, im_clus_num_pos_d);
    }

    int max_level = isp_levels - 1;
    for (int i = max_level; i >= min_level; --i)
    {
        // std::cout << "start isp level: " << i << std::endl;
        switch_level(i);

        if (i == max_level)
        {
            cudaMemset(msg_d, 0, sizeof(float) * num_bd * K);
            cudaMemset(belief_d, 0, sizeof(float) * num_clus * K);
        }
        else
            hbp_g(plane_d, belief_d);
    }

    switch_level(min_level);
    auto results_img = map_msg(plane_d, belief_d);
    std::cout << "inference done" << std::endl;
    return results_img;
}

std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>>
HBP_ISP::run_ms_vis(const int min_level)
{
    std::cout << "start HBP-ISP" << std::endl;
    std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> results;
    for (int i = 0; i < isp_levels; ++i)
    {
        switch_level(i);
        reverse_bd_id(bd_d, reverse_id_d);
        compute_dist_color_pos(mean_d, dc_d);
        sort_clus(clus_sort_d, clus_num_pos_d, im_clus_num_pos_d);
    }

    int max_level = isp_levels - 1;
    for (int i = max_level; i >= min_level; --i)
    {
        std::cout << "isp level: " << i << std::endl;
        switch_level(i);

        if (i == max_level)
        {
            cudaMemset(msg_d, 0, sizeof(float) * num_bd * K);
            cudaMemset(belief_d, 0, sizeof(float) * num_clus * K);
        }
        else
            hbp_g(plane_d, belief_d);

        auto disp_label = map_msg(plane_d, belief_d);
        std::cout << "map msg done" << std::endl;
        cv::Mat spx_img = draw_spx_mean(mean_d);
        results.emplace_back(std::make_tuple(disp_label.first, disp_label.second, spx_img));
    }

    std::cout << "inference done" << std::endl;
    return results;
}
