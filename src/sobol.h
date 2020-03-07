#pragma once
#include <cstdlib> // *** Thanks to Leonhard Gruenschloss and Mike Giles   ***
#include <cmath>   // *** for pointing out the change in new g++ compilers ***

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <vector>
#include <assert.h>
#include <boost/math/distributions/normal.hpp>
#include <Eigen/Core>

inline double **sobol_points(unsigned N, unsigned D, const char *dir_file)
{
  std::ifstream infile(dir_file,std::ios::in);
  if (!infile) {
    std::cout << "Input file containing direction numbers cannot be found!\n";
    exit(1);
  }
  char buffer[1000];
  infile.getline(buffer,1000,'\n');
  
  // L = max number of bits needed 
  unsigned L = (unsigned)ceil(log((double)N)/log(2.0)); 

  // C[i] = index from the right of the first zero bit of i
  unsigned *C = new unsigned [N];
  C[0] = 1;
  for (unsigned i=1;i<=N-1;i++) {
    C[i] = 1;
    unsigned value = i;
    while (value & 1) {
      value >>= 1;
      C[i]++;
    }
  }
  
  // POINTS[i][j] = the jth component of the ith point
  //                with i indexed from 0 to N-1 and j indexed from 0 to D-1
  double **POINTS = new double * [N];
  for (unsigned i=0;i<=N-1;i++) POINTS[i] = new double [D];
  for (unsigned j=0;j<=D-1;j++) POINTS[0][j] = 0; 

  // ----- Compute the first dimension -----
  
  // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
  unsigned *V = new unsigned [L+1]; 
  for (unsigned i=1;i<=L;i++) V[i] = 1 << (32-i); // all m's = 1

  // Evalulate X[0] to X[N-1], scaled by pow(2,32)
  unsigned *X = new unsigned [N];
  X[0] = 0;
  for (unsigned i=1;i<=N-1;i++) {
    X[i] = X[i-1] ^ V[C[i-1]];
    POINTS[i][0] = (double)X[i]/pow(2.0,32); // *** the actual points
    //        ^ 0 for first dimension
  }
  
  // Clean up
  delete [] V;
  delete [] X;
  
  
  // ----- Compute the remaining dimensions -----
  for (unsigned j=1;j<=D-1;j++) {
    
    // Read in parameters from file 
    unsigned d, s;
    unsigned a;
    infile >> d >> s >> a;
    unsigned *m = new unsigned [s+1];
    for (unsigned i=1;i<=s;i++) infile >> m[i];
    
    // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
    unsigned *V = new unsigned [L+1];
    if (L <= s) {
      for (unsigned i=1;i<=L;i++) V[i] = m[i] << (32-i); 
    }
    else {
      for (unsigned i=1;i<=s;i++) V[i] = m[i] << (32-i); 
      for (unsigned i=s+1;i<=L;i++) {
	V[i] = V[i-s] ^ (V[i-s] >> s); 
	for (unsigned k=1;k<=s-1;k++) 
	  V[i] ^= (((a >> (s-1-k)) & 1) * V[i-k]); 
      }
    }
    
    // Evalulate X[0] to X[N-1], scaled by pow(2,32)
    unsigned *X = new unsigned [N];
    X[0] = 0;
    for (unsigned i=1;i<=N-1;i++) {
      X[i] = X[i-1] ^ V[C[i-1]];
      POINTS[i][j] = (double)X[i]/pow(2.0,32); // *** the actual points
      //        ^ j for dimension (j+1)
   }
    
    // Clean up
    delete [] m;
    delete [] V;
    delete [] X;
  }
  delete [] C;
  
  return POINTS;
}

inline double **sobol_points(unsigned N, unsigned D) {
  const char *dir_file = "../sobol_data/new-joe-kuo-6.21201";
  return sobol_points(N, D, dir_file);
}

inline Eigen::MatrixXd sobol_points_normal(unsigned N, unsigned D, int k) {
    const auto uniform = sobol_points(N+k, D);
    boost::math::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd ret(N, D);
    for (int i =0; i< N; ++i) {
      for (int j = 0; j < D; ++j) {
        double z = uniform[i+k][j];
        ret(i, j) = quantile(dist, z);
      }
    } 
    return ret;
}