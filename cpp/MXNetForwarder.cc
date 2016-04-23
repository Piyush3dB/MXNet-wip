

#include <stdio.h>
#include <assert.h>
#include <stdint.h>

// Path for c_predict_api
#include <c_predict_api.h>
#include <MXNetForwarder.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>




    /* Constructor */
MXNetForwarder::MXNetForwarder(int w, int h, int c, const char* SymbolJson, const char* NetParams, int paramLen){
    
    // Image dimenstions and size used during forwarding
    this->image_size = w*h*c;
    
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




