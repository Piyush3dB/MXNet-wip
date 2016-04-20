/*!
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 */

//
//  File: image-classification-predict.cpp
//  This is a simple predictor which shows
//  how to use c api for image classfication
//

#include <stdio.h>
#include <assert.h>
#include <stdint.h>

// Path for c_predict_api
#include <c_predict_api.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>


// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            assert(false);
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    const char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        delete[] buffer_;
        buffer_ = NULL;
    }
};



void SortOutputResult(const std::vector<float>& data, std::multimap<int,int> &resultsMap) {
    
    int pctg = 0;

    // Insert into multimap
    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
        pctg = (int)(10000*data[i]);
        resultsMap.insert ( std::pair<int,int>(pctg,i) );
    }

}


void PrintOutputResult(const std::multimap<int,int>& resultsMap, const std::vector<std::string>& synset) {
    
        std::cout << "\n\nTop 5 predictions:\n";
        auto it = resultsMap.rbegin();
        int i = 0;
        float pctgFlt = 0;
        for (i=0; i<5; i++){
            pctgFlt = (float)((*it).first)/100;
            printf("%3.3f => %s \n", pctgFlt, synset[(*it).second].c_str());
            it++;
        }


}



    // LoadSynsets
std::vector<std::string> LoadSynset(const char *filename) {
    std::ifstream fi(filename);

    if ( !fi.is_open() ) {
        std::cerr << "Error opening file " << filename << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while ( fi >> synset ) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}






// MXNet forwarder class
class MXNetForwarder {
  public:

    /* Handler context for predictor */
    PredictorHandle pCtx = 0;

    /* Json string */
    std::string SymbolJson;

    /* Image dimension */
    int image_size = 0;


    /* Constructor */
    MXNetForwarder(int w, int h, int c){

        // Image dimenstions and size
        this->image_size = w*h*c;

        // Models path for your model, you have to modify it
        BufferFile json_data( "../../MXNetModels/cifar1000VGGmodel/Inception_BN-symbol.json");
        BufferFile param_data("../../MXNetModels/cifar1000VGGmodel/Inception_BN-0039.params");

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
        MXPredCreate((const char*)json_data.GetBuffer(),
                     (const char*)param_data.GetBuffer(),
                     static_cast<size_t>(param_data.GetLength()),
                     1,
                     0,
                     1,
                     input_keys,
                     input_shape_indptr,
                     input_shape_data,
                     &this->pCtx);

    }

    void Forward(std::vector<mx_float> image_data){

        //-- Set Input Image
        MXPredSetInput(this->pCtx, "data", image_data.data(), this->image_size);

        //-- Do Predict Forward
        MXPredForward(this->pCtx);
    }


    void GetOutput(){
        
        //-- Get Output shape and size
        mx_uint output_index = 0;
        mx_uint *shape = 0;
        mx_uint shape_len;
        MXPredGetOutputShape(this->pCtx, output_index, &shape, &shape_len);
        size_t size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) {
           size *= shape[i];   
        }

        //-- Get Output result
        std::vector<float> data(size);
        MXPredGetOutput(this->pCtx, output_index, &(data[0]), size);

        // Synset path for your model, you have to modify it
        std::vector<std::string> synset = LoadSynset("../../MXNetModels/cifar1000VGGmodel/synset.txt");

        //-- Sort output result
        std::multimap<int,int> resultsMap;
        SortOutputResult(data, resultsMap);
        
        //-- Print Output Data
        PrintOutputResult(resultsMap, synset);
    
    }


    void Free() {
        // Release Predictor
        MXPredFree(this->pCtx);
    }


};




int main(int argc, char* argv[]) {

    std::cout << "\nHere in main()...\n" << std::endl;



///////////////////////
    /* Read cat data */
    std::ifstream is ("cat_224x224x3.bin", std::ifstream::binary);
    // get length of file:
    is.seekg (0, is.end);
    int length = is.tellg();
    is.seekg (0, is.beg);

    uint8_t * imAsCol = new uint8_t [length];

    std::cout << "Reading " << length << " characters... ";
    // read data as a block:
    is.read(reinterpret_cast<char *>(imAsCol), length);

    if (is)
      std::cout << "all characters read successfully. " << is.gcount() << " read" ;
    else
      std::cout << "error: only " << is.gcount() << " could be read";
    is.close();

    // ...imAsCol contains the entire file...
    // Adjust to the mean image
    mx_float meanAdjust = (mx_float) 120;
    int image_size = 224 * 224 * 3;
    std::vector<mx_float> image_data = std::vector<mx_float>(image_size);
    mx_float* image_data_ptr = image_data.data();
    
    for (int j = 0; j < image_size; j++) {
        image_data_ptr[j] = (mx_float)imAsCol[j] - meanAdjust;
    }
    std::cout << '\n';
///////////////////////


    std::cout << "Constructor...\n";
    MXNetForwarder mxObj(224, 224, 3);

    std::cout << "Forward data...\n";
    mxObj.Forward(image_data);


    std::cout << "Retireve output...\n";
    mxObj.GetOutput();

    std::cout << "Freeing...\n";
    mxObj.Free();

    delete[] imAsCol;

    return 0;
}


/*
Should see in console:


Top 5 predictions:
39.470 =>  tiger cat 
22.070 =>  tabby, tabby cat 
20.210 =>  Egyptian cat 
12.200 =>  lynx, catamount 
0.620 =>  Persian cat 



*/