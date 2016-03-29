/* $Id: gigrnd_par.cpp 3606 2014-02-26 16:26:46Z johanl $ */

#include "gig_par.h"

#include <chrono>
#include <mex.h>
#include <random>

#ifdef _OPENMP
#include<omp.h>
#endif

/**Checks if a given array contains a non-sparse, real valued, double matrix.
 *@param tmp pointer to a matlab array.*/
bool mxIsRealDoubleArray(const mxArray* tmp){
  return !mxIsEmpty(tmp) && mxIsDouble(tmp) && !mxIsSparse(tmp) &&
    !mxIsComplex(tmp); 
}

/**Checks if a given array contains a double scalar.
 *@param tmp pointer to a matlab array.*/
bool mxIsRealScalar(const mxArray* tmp){
  return !mxIsEmpty(tmp) && mxIsDouble(tmp) && !mxIsComplex(tmp) &&
    mxGetNumberOfElements(tmp)==1;
}

/**Checks if a given array contains a positive double scalar.
 *@param tmp pointer to a matlab array.*/
bool mxIsRealPosScalar(const mxArray* tmp){
  return mxIsRealScalar(tmp) && mxGetScalar(tmp)>0;
}

/**Extracts the number of threads requested for openMP.
 *@param nrhs,prhs inputs to the main mex-file
 *@param i an integer telling which argument to consider.
 */
int mxFindNthreads(int nrhs, const mxArray *prhs[], int i){
#ifdef _OPENMP
  int Nthreads = omp_get_num_procs();
#else
  int Nthreads = 1;
#endif

  //use the )(i+1)th input (recall zero index) - nthreads
  if(nrhs>i && !mxIsEmpty(prhs[i])){
    if( !mxIsRealPosScalar(prhs[i]) ){
      mexErrMsgIdAndTxt("mex_error_checking:badinput",
                        "The %u argument should be either empty=[]; or a positive scalar",
                        i+1);
    }
    Nthreads = static_cast<int>( mxGetScalar(prhs[i]) );
    //ensure that Nthreads>1
    if(Nthreads<1){ Nthreads=1; }
  }
  return Nthreads;
}//int mxFindNthreads

#ifdef _OPENMP
/*Function that waits 1/100 of a second at clear mex, reduces risk of segfault
  when using openMP
 */
static void cleanupOMP(void){
  //wait for 1/100 s, to avoid a segfault race condition in clear mex.
  for(double end_t = omp_get_wtime()+0.01; omp_get_wtime()<end_t; ){}
}
#endif


using std::vector;

uint_fast64_t sample_gig(double** ptr_VEC, double* VEC, double* x_ptr,
                         size_t n, uint_fast64_t seed, bool debug){
  //find maximum number of threads
	#ifdef _OPENMP
  int Nthreads = omp_get_max_threads();
	#else
  int Nthreads = 1;
	#endif
  
  //create a random generator
  if( seed==0 ){
    //do we need a random seed?
    using namespace std::chrono;
    auto Depoch_ns = duration_cast<nanoseconds>
      ( high_resolution_clock::now().time_since_epoch() );
    //use nanoseconds since epoch as seed
    seed = static_cast<uint_fast64_t>( Depoch_ns.count() );
  }
  //create random-generator
  std::mt19937_64 rgen(seed);

  //vector of the iterations - used for debuging
  vector<unsigned int> it(Nthreads,0);

  //vector of different gig-objects (one per thread)
  vector<gig> gigstream;
  gigstream.reserve(Nthreads);
  for(int i=0; i<Nthreads; ++i){
    if(debug){
      gigstream.push_back( gig(seed) );
    }else{
      gigstream.push_back( gig( static_cast<uint_fast64_t>(rgen()) ) );
    }
  }

  //parameters, uses fixed value if ptr_VEC[i]==nullptr
  double p = VEC[0];
  double a = VEC[1];
  double b = VEC[2];

  //use firstprivate to obtain thread-private copies that are initialised to the
  //values existing before the thread.
#pragma omp parallel for firstprivate(p,a,b)
  for(size_t i=0; i<n; ++i){
    //which thread are we in? (0 for non-openMP code)
#ifdef _OPENMP
    int cur_thread = omp_get_thread_num();
#else
    int cur_thread = 0;
#endif
    
    //extract current parameter values (unless constant)
    if( ptr_VEC[0]!=nullptr ){ p = ptr_VEC[0][i]; }
    if( ptr_VEC[1]!=nullptr ){ a = ptr_VEC[1][i]; }
    if( ptr_VEC[2]!=nullptr ){ b = ptr_VEC[2][i]; }
    x_ptr[i] = gigstream[cur_thread].sample(p, a, b);

    //which thread and iteration are we in? (debug use)
    if(debug){
      x_ptr[i+n] = cur_thread;
      x_ptr[i+2*n] = it[cur_thread]++;
    }
	}//for(size_t i=0; i<n; ++i)
  
  return seed;
}//uint_fast64_t sample_gig


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  //make sure that seed is a 64-bit number
  uint_fast64_t seed = 0;
  
  //x = gigrnd_par(p, a, b, seed, nthreads)
  //p, a, b matrices with the same number of elements (or scalars)
  //seed matrix with 1 element (0 uses time since epoch as seed)
  if( nlhs>1 ){
    mexErrMsgIdAndTxt("rgig_alt:badinput", "Function returns ONE value.");
  }
  if(nrhs<3 || nrhs>6){
    mexErrMsgIdAndTxt("rgig_alt:badinput",
                      "Function requires 3-6 input(s), %i found.", nrhs);
  }
  
  //check that all three inputs are matrices and extract the sizes
  mwSize sz[3];
  mwSize maxSize=0;
  size_t iMax=0;
  for(size_t i=0; i<3; ++i){
    if( !mxIsRealDoubleArray(prhs[i]) ){
      mexErrMsgIdAndTxt("rgig_alt:badinput",
                        "Argument %i should be a matrix", i+1);
    }
    sz[i] = mxGetNumberOfElements(prhs[i]);
    if( sz[i]>maxSize ){
      maxSize=sz[i];
      iMax=i;
    };
  }
  //check that all three inputs have the same size (or are scalar)
  for(size_t i=0; i<3; ++i){
    if(sz[i]!=maxSize && sz[i]!=1){
      mexErrMsgIdAndTxt("rgig_alt:badinput",
                        "Argument %i should either have 1 or %i elements", i+1, maxSize);
    }
  }

  //fourth input, the seed
  if(nrhs>3 && !mxIsEmpty(prhs[3])){
    if( !mxIsRealScalar(prhs[3]) ){
      mexErrMsgIdAndTxt("rgig_alt:badinput",
                        "Fourth argument should be either empty=[]; or a scalar");
    }
    if( mxGetScalar(prhs[3])<0 ){
      mexErrMsgIdAndTxt("rgig_alt:badinput",
                        "Fourth argument should non-negative, is %f",
                        mxGetScalar(prhs[3]));
    }
    seed = static_cast<uint_fast64_t>( mxGetScalar(prhs[3]) );
  }//if(nrhs>3 && !mxIsEmpty(prhs[3]))

#ifdef _OPENMP
  mexAtExit( cleanupOMP );
  //fifth input - nthreads
  int Nthreads = mxFindNthreads(nrhs, prhs, 4);
  //set nbr threads in openMP
  omp_set_dynamic(1);
  omp_set_num_threads(Nthreads);
#endif

  //sixth (undocumented) debug input
  bool debug = false;
  if(nrhs>5 && !mxIsEmpty(prhs[5])){
    if( !mxIsRealScalar(prhs[5]) ){
      mexErrMsgIdAndTxt("rgig_alt:badinput",
                        "Sixth argument should be either empty=[]; or a scalar");
    }
    debug = (mxGetScalar(prhs[5])==0);
  }
  
  
  //extract pointers to the data
	double* ptr_VEC[3];
  //or extract the scalars.
  double VEC[3];
  for(size_t i=0; i<3; ++i){
    if(sz[i]==maxSize){
      ptr_VEC[i] = mxGetPr( prhs[i] );
      VEC[i] = ptr_VEC[i][0];
    }else{
      //sz[i] must be either maxSize of a scalar, if it's a scalar extract value
      //and set pointer to nullptr
      VEC[i] = mxGetScalar(prhs[i]);
      ptr_VEC[i] = nullptr;
    }
  }//for(int i=0; i<3; ++i)
  
  //allocate memory for return data, same size as largest input (first if
  //several at tied for largest)
  if(debug){
    plhs[0] = mxCreateNumericMatrix(maxSize, 3, mxDOUBLE_CLASS, mxREAL);
  }else{
    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[iMax]),
                                   mxGetDimensions(prhs[iMax]),
                                   mxDOUBLE_CLASS, mxREAL);
  }
  double* x_ptr = mxGetPr( plhs[0] );

  //sample from the gig_distribution
  seed = sample_gig(ptr_VEC, VEC, x_ptr, static_cast<size_t>(maxSize),
                    seed, debug);
  
  if(debug){
    mexPrintf("Sanity check of inputs:\n");
    mexPrintf("  # elements: ");
    for(size_t i=0; i<3; ++i){ mexPrintf("%llu, ", sz[i]); }
    mexPrintf("\n     maxSize: %llu", maxSize);
    mexPrintf("\n        iMax: %llu", iMax);
    mexPrintf("\n        seed: %lli", seed);
    mexPrintf("\n       debug: %u\n", debug);
  
#ifdef _OPENMP
    mexPrintf("\nNumber of threads: %i/%i\n", Nthreads, omp_get_max_threads());
#endif
  }

}//extern "C" void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])

//COMPILE FOR WINDOWS (assuming cygwin install with openMP):
//mex -largeArrayDims gigrnd_par.cpp gig_par.cpp COMPFLAGS="$COMPFLAGS -std=c++0x -Wall -fopenmp"..
//GM_ADD_LIBS="$GM_ADD_LIBS -lgomp"

//COMPILE FOR LINUX (gcc 4.6.2 or greater to ensure C++0x, i.e. <random.h>)
//mex -largeArrayDims gigrnd_par.cpp gig_par.cpp CXXOPTIMFLAGS="-O3 -march=native" ...
//CXXFLAGS="\$CXXFLAGS -std=c++0x -Wall -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
