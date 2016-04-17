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

 #define IMAGE_BYTES (224*224*3)

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
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        delete[] buffer_;
        buffer_ = NULL;
    }
};

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

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
        printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if ( data[i] > best_accuracy ) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
    synset[best_idx].c_str(), best_idx, best_accuracy);
}

int main(int argc, char* argv[]) {

    std::cout << "\nHere in main()...\n" << std::endl;

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
        //printf("imAsCol[%d] = %d. %f\n", j, (uint8_t)imAsCol[j], (mx_float)image_data_ptr[j]);
    }


    //for (auto i = image_data.begin(); i != image_data.end(); ++i) {
#if 0
    for (auto const& i : image_data) {
        std::cout << i << ' ';
    }
#endif
    std::cout << '\n';


    // Models path for your model, you have to modify it
    BufferFile json_data( "../../MXNetModels/cifar1000VGGmodel/Inception_BN-symbol.json");
    BufferFile param_data("../../MXNetModels/cifar1000VGGmodel/Inception_BN-0039.params");

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;

    // Image size and channels
    int width = 224;
    int height = 224;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    // ( trained_width, trained_height, channel, num)
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(width),
                                        static_cast<mx_uint>(height) };
    PredictorHandle pCtx = 0;  // alias for void *

    //-- Create Predictor
    MXPredCreate((const char*)json_data.GetBuffer(),
                 (const char*)param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pCtx);

    //-- Set Input Image
    MXPredSetInput(pCtx, "data", image_data.data(), image_size);


    //-- Do Predict Forward
    MXPredForward(pCtx);

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    //-- Get Output Result
    MXPredGetOutputShape(pCtx, output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) {
       size *= shape[i];   
    }

    std::vector<float> data(size);

    MXPredGetOutput(pCtx, output_index, &(data[0]), size);

    // Release Predictor
    MXPredFree(pCtx);

    // Synset path for your model, you have to modify it
    std::vector<std::string> synset = LoadSynset("../../MXNetModels/cifar1000VGGmodel/synset.txt");

    //-- Print Output Data
    PrintOutputResult(data, synset);

    delete[] imAsCol;

    return 0;
}
