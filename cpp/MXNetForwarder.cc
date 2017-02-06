// Path for c_predict_api
#include <c_predict_api.h>
#include <c_api.h>
#include <MXNetForwarder.h>

    /* Constructor */
MXNetForwarder::MXNetForwarder(int w, int h, int c, const char* SymbolJson, const char* NetParams, int paramLen){
    
    // Image dimenstions and size used during forwarding
    this->image_size = w*h*c;
    this->SymbolJson = SymbolJson;
    

    
    // Parameters
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;
    const mx_uint input_shape_indptr[2] = { 0, 4 };
    // ( trained_width, trained_height, channel, num)
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(c),
                                        static_cast<mx_uint>(w),
                                        static_cast<mx_uint>(h) };
    
    // Create Predictor
    MXPredCreate(SymbolJson,
                 NetParams,
                 paramLen,
                 1,
                 0,
                 1,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &this->pCtx);
    
    // Create symbol from JSON
    MXSymbolCreateFromJSON(this->SymbolJson, &this->handle);
}


void MXNetForwarder::InferShape(
    const std::map<std::string, std::vector<mx_uint> > &arg_shapes,
    std::vector<std::vector<mx_uint> > *in_shape,
    std::vector<std::vector<mx_uint> > *aux_shape,
    std::vector<std::vector<mx_uint> > *out_shape) const {

  std::vector<const char *> keys;
  std::vector<mx_uint> arg_ind_ptr;
  std::vector<mx_uint> arg_shape_data;

  for (const auto &arg : arg_shapes) {
    keys.push_back(arg.first.c_str());
    arg_ind_ptr.push_back(arg_shape_data.size());
    for (auto i : arg.second) {
      arg_shape_data.push_back(i);
    }
  }
  arg_ind_ptr.push_back(arg_shape_data.size());

  mx_uint in_shape_size;
  const mx_uint *in_shape_ndim;
  const mx_uint **in_shape_data;
  mx_uint out_shape_size;
  const mx_uint *out_shape_ndim;
  const mx_uint **out_shape_data;
  mx_uint aux_shape_size;
  const mx_uint *aux_shape_ndim;
  const mx_uint **aux_shape_data;
  int complete;

  MXSymbolInferShape(this->handle, keys.size(), keys.data(),
                     arg_ind_ptr.data(), arg_shape_data.data(),
                     &in_shape_size, &in_shape_ndim, &in_shape_data,
                     &out_shape_size, &out_shape_ndim, &out_shape_data,
                     &aux_shape_size, &aux_shape_ndim, &aux_shape_data,
                     &complete);

  if (complete) {
    for (mx_uint i = 0; i < in_shape_size; ++i) {
      in_shape->push_back(std::vector<mx_uint>());
      for (mx_uint j = 0; j < in_shape_ndim[i]; ++j) {
        (*in_shape)[i].push_back(in_shape_data[i][j]);
      }
    }
    for (mx_uint i = 0; i < aux_shape_size; ++i) {
      aux_shape->push_back(std::vector<mx_uint>());
      for (mx_uint j = 0; j < aux_shape_ndim[i]; ++j) {
        (*aux_shape)[i].push_back(aux_shape_data[i][j]);
      }
    }
    for (mx_uint i = 0; i < out_shape_size; ++i) {
      out_shape->push_back(std::vector<mx_uint>());
      for (mx_uint j = 0; j < out_shape_ndim[i]; ++j) {
        (*out_shape)[i].push_back(out_shape_data[i][j]);
      }
    }
  }
}


#if 0

void MXNetForwarder::InferShape(){


        unsigned int   arg_shape_size;  // = mx_uint()
        unsigned int*  arg_shape_ndim;  // = ctypes.POINTER(mx_uint)()
        unsigned int** arg_shape_data;  // = ctypes.POINTER(ctypes.POINTER(mx_uint))()

        unsigned int   out_shape_size;  // = mx_uint()
        unsigned int*  out_shape_ndim;  // = ctypes.POINTER(mx_uint)()
        unsigned int** out_shape_data;  // = ctypes.POINTER(ctypes.POINTER(mx_uint))()

        unsigned int   aux_shape_size;  // = mx_uint()
        unsigned int*  aux_shape_ndim;  // = ctypes.POINTER(mx_uint)()
        unsigned int** aux_shape_data;  // = ctypes.POINTER(ctypes.POINTER(mx_uint))()

        int complete; // = ctypes.c_int()

        int retval;

        retval = MXSymbolInferShape(
            this->handle,

            mx_uint(len(indptr) - 1),
            c_array(ctypes.c_char_p, keys),
            c_array(mx_uint, indptr),
            c_array(mx_uint, sdata),
            
            &arg_shape_size,
            &arg_shape_ndim,
            &arg_shape_data,
            &out_shape_size,
            &out_shape_ndim,
            &out_shape_data,
            &aux_shape_size,
            &aux_shape_ndim,
            &aux_shape_data,
            &complete);


    MXNET_DLL int MXSymbolInferShape(
                                 this->SymbolJson, //SymbolHandle sym,
                                 
                                 mx_uint num_args,
                                 const char** keys,
                                 const mx_uint *arg_ind_ptr,
                                 const mx_uint *arg_shape_data,

                                 mx_uint *in_shape_size,
                                 const mx_uint **in_shape_ndim,
                                 const mx_uint ***in_shape_data,
                                 mx_uint *out_shape_size,
                                 const mx_uint **out_shape_ndim,
                                 const mx_uint ***out_shape_data,
                                 mx_uint *aux_shape_size,
                                 const mx_uint **aux_shape_ndim,
                                 const mx_uint ***aux_shape_data,
                                 int *complete);

}
#endif





void MXNetForwarder::Forward(std::vector<mx_float> image_data){

    //-- Set Input Image
    MXPredSetInput(this->pCtx, "data", image_data.data(), this->image_size);

    //-- Do Predict Forward
    MXPredForward(this->pCtx);
}


std::multimap<int,int> MXNetForwarder::SortOutputResult(const std::vector<float>& data) {
    
    int pctg = 0;
    std::multimap<int,int> resultsMap;
    
    // Insert into multimap
    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
        pctg = (int)(10000*data[i]);
        resultsMap.insert ( std::pair<int,int>(pctg,i) );
    }
    return resultsMap;
}


std::multimap<int,int> MXNetForwarder::GetOutput(){
    
    //-- Get Output shape and size
    mx_uint *shape = 0;
    mx_uint shape_len;
    MXPredGetOutputShape(this->pCtx, 0, &shape, &shape_len);
    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) {
       size *= shape[i];   
    }

    printf("Size %d. Shape %d: [%d, %d]\n", size, shape_len, shape[0], shape[1]);
    
    //-- Get Output result
    std::vector<float> data(size);
    MXPredGetOutput(this->pCtx, 0, &(data[0]), size);
    
    //-- Sort output result according to probability values
    auto resultsMap = SortOutputResult(data);
    
    return resultsMap;
}


void MXNetForwarder::Free() {
    //-- Release Predictor context
    MXPredFree(this->pCtx);
}




////

