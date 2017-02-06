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
#include "c_predict_api.h"
#include "MXNetForwarder.h"

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






void PrintOutputResult(const std::multimap<int,int>& resultsMap, const std::vector<std::string>& synset) {
    std::cout << "\nTop 5 predictions:\n";
    auto it = resultsMap.rbegin();
    int i = 0;
    float pctgFlt = 0;
    for (i=0; i<10; i++){
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








std::vector<mx_float> LoadImage(std::string imgFname, int imgSize){

    /* Read image data */
    std::ifstream is (imgFname, std::ifstream::binary);

    uint8_t * imAsCol = new uint8_t [imgSize];

    // read data as a block:
    is.read(reinterpret_cast<char *>(imAsCol), imgSize);

    // Adjust to the mean image
    mx_float meanAdjValue = (mx_float) 120;
    std::vector<mx_float> adjImage = std::vector<mx_float>(imgSize);
    mx_float* adjImage_ptr = adjImage.data();
    
    for (int j = 0; j < imgSize; j++) {
        adjImage_ptr[j] = (mx_float)imAsCol[j] - meanAdjValue;
    }

    delete[] imAsCol;

    return adjImage;
}





int main(int argc, char* argv[]) {

    //-- Load the input image
    std::cout << "\nLoading image...\n";
    auto image_data = LoadImage("cat_224x224x3.bin", (224*224*3));

    //-- Load MXNet network and parameters
    std::cout << "\nLoading network parameters...\n";
    BufferFile json_data( "../../MXNetModels/newModel/model/Inception_BN-symbol.json");
    BufferFile param_data("../../MXNetModels/newModel/model/Inception_BN-0039.params");

    //-- Create Forwarder context
    std::cout << "Constructor...\n";
    auto symb   = (const char *) json_data.GetBuffer();
    auto param  = (const char *) param_data.GetBuffer();
    auto nParam = static_cast<size_t>(param_data.GetLength());
    MXNetForwarder mxObj(224, 224, 3, symb, param, nParam);

    //-- Forward the image throught the network
    std::cout << "Forward data...\n";
    mxObj.Forward(image_data);

    //-- Retireve output probability map
    std::cout << "Retireve output...\n";
    auto resultsMap = mxObj.GetOutput();

    //-- Release predictor context
    std::cout << "Freeing...\n";
    mxObj.Free();

    //-- Display results
    std::cout << "Sort and display predictions...\n";
    auto synset = LoadSynset("../../MXNetModels/newModel/model/synset.txt");
    PrintOutputResult(resultsMap, synset);

    return 0;
}


/*
Should see in console:


Top 5 predictions:
29.710 =>  tiger cat 
26.580 =>  lynx, catamount 
19.800 =>  Egyptian cat 
17.770 =>  tabby, tabby cat 
2.610 =>  kit fox, Vulpes macrotis 
0.460 =>  Siamese cat, Siamese 
0.420 =>  red fox, Vulpes vulpes 
0.200 =>  Persian cat 
0.170 =>  plastic bag 
0.130 =>  window screen 

*/