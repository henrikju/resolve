double sinc(double x);

void grid_complex(unsigned int Nk, unsigned int Nq,
                  double complex grid[Nk][Nq],
                  double vk, double vq,
                  unsigned int Nuv,
                  double complex inpoints[Nuv],
                  double u[Nuv], double v[Nuv],
                  unsigned int precision);


void grid_abs_squared(unsigned int Nk, unsigned int Nq,
                      double grid[Nk][Nq],
                      double vk, double vq,
                      unsigned int Nuv,
                      double inpoints[Nuv],
                      double u[Nuv], double v[Nuv],
                      unsigned int precision);


void grid_Martinc(unsigned int Nk, unsigned int Nq,
                  double complex grid[Nk][Nq],
                  double vk, double vq,
                  unsigned int Nuv,
                  double complex inpoints[Nuv],
                  double u[Nuv], double v[Nuv],
                  unsigned int precision);