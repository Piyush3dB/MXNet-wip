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
    
    //-- Create Predictor
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
    
}

void MXNetForwarder::InferShape(){

        MXSymbolCreateFromJSON(this->SymbolJson, &this->handle);

#if 0

        arg_shape_size;
        arg_shape_ndim = ctypes.POINTER(mx_uint)()
        arg_shape_data = ctypes.POINTER(ctypes.POINTER(mx_uint))()
        out_shape_size = mx_uint()
        out_shape_ndim = ctypes.POINTER(mx_uint)()
        out_shape_data = ctypes.POINTER(ctypes.POINTER(mx_uint))()
        aux_shape_size = mx_uint()
        aux_shape_ndim = ctypes.POINTER(mx_uint)()
        aux_shape_data = ctypes.POINTER(ctypes.POINTER(mx_uint))()


    MXNET_DLL int MXSymbolInferShape(this->SymbolJson, //SymbolHandle sym,
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

#endif

}




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

