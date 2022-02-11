#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <cstdint>

namespace py = pybind11;
using namespace std;

void julia(py::array_t<double>& data, int width, int height, int zoom, double cX, double cY, double moveX, double moveY, int maxIter)
{
  auto dat = data.mutable_unchecked<2>();
  #pragma omp parallel for
  for (int x = 0; x < width; x++){
    for (int y = 0; y < height; y++){
      double zx = 1.5*(x - width/2)/(0.5*zoom*width) + moveX; 
      double zy = 1.0*(y - height/2)/(0.5*zoom*height) + moveY; 
      int i = maxIter; 
      double tmp; 
      while((zx*zx + zy*zy < 4) && (i > 1)){
        tmp = zx*zx - zy*zy + cX; 
        zy = 2.0*zx*zy + cY; 
        zx = tmp; 
        i -= 1; 
      }  
      dat(x,y) = (i << 21) + (i << 10) + i*8; 
    }
  }
}

PYBIND11_MODULE(ijulia,m){
  m.doc() = "pybind11 wrap for juliaset";
  m.def("julia", &julia);
}
