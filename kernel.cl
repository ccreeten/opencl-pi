  __kernel void pi(
      const long steps_total,
      const long steps_per_kernel,
      const double step,
      __global double *global_sums)
  {
      const int global_id  = get_global_id(0);
      const int local_id   = get_local_id(0);
      const int local_size = get_local_size(0);
      const double16 unit = (double16)1.0;

      const long vector_size = 16;
      const double16 deltas  = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

      const long work_dx_start     = global_id * steps_per_kernel;
      const long work_dx_end       = min(work_dx_start + steps_per_kernel, steps_total);
      const long vectorized_dx_end = work_dx_end / vector_size * vector_size;

      double work_sum = 0.0;
      for (long dx = work_dx_start; dx < vectorized_dx_end; dx += vector_size)
      {
          const double16 mid_points   = (dx - 0.5 + deltas) * step;
          const double16 partial_sums = (4.0 / (1.0 + mid_points * mid_points));

          const double8 d8 = partial_sums.lo + partial_sums.hi;
          const double4 d4 = d8.lo + d8.hi;
          const double2 d2 = d4.lo + d4.hi;

          work_sum += d2.lo + d2.hi;
      }

      for (long dx = vectorized_dx_end; dx < work_dx_end; dx++)
      {
          const double mid_point = (dx - 0.5) * step;
          work_sum += 4.0 / (1.0 + mid_point * mid_point);
      }

      const double group_sum = work_group_reduce_add(work_sum);
      if (local_id == 0)
      {
          global_sums[global_id / local_size] = group_sum;
      }
  }

