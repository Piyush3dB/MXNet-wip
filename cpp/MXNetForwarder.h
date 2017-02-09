
#ifndef MXNET_FORWARDER_H_
#define MXNET_FORWARDER_H_
// Path for c_predict_api
#include <c_predict_api.h>
#include <c_api.h>

#include <vector>
#include <map>


// MXNet forwarder class
class MXNetForwarder {
  public:

    /* Handler context for predictor */
    PredictorHandle pCtx = nullptr;

    /* Json symbol string */
    char* SymbolJson;

    /* Network parameters */
    const char* NetParams;

    /* Image dimension */
    int image_size = 0;

    /* Handle for symbol */
    void* handle;


    MXNetForwarder(int w, int h, int c, char* SymbolJson, const char* NetParams, int paramLen);

    void Forward(std::vector<mx_float> image_data);
    
      void InferShape() const;

    std::multimap<int,int> SortOutputResult(const std::vector<float>& data);

    std::multimap<int,int> GetOutput();

    void Free();

};




#endif  // MXNET_FORWARDER_H_

